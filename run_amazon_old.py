import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import matplotlib.pyplot as plt
import torch.optim as optim

from src.dataloader_pnet import parse_pnet, ListToTorch
from srcOld import log

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
from src import networks
from src import pnetProcess
from src import utils

# # load training data
# Aind = torch.load('../data/casp11/AminoAcidIdx.pt')
# Yobs = torch.load('../data/casp11/RCalpha.pt')
# MSK  = torch.load('../data/casp11/Masks.pt')
# S     = torch.load('../data/casp11/PSSM.pt')
# # load validation data
# AindVal = torch.load('../data/casp11/AminoAcidIdxVal.pt')
# YobsVal = torch.load('../data/casp11/RCalphaVal.pt')
# MSKVal  = torch.load('../data/casp11/MasksVal.pt')
# SVal     = torch.load('../data/casp11/PSSMVal.pt')
#
# # load Testing data
# AindTesting = torch.load('../data/casp11/AminoAcidIdxTesting.pt')
# YobsTesting = torch.load('../data/casp11/RCalphaTesting.pt')
# MSKTesting  = torch.load('../data/casp11/MasksTesting.pt')
# STesting     = torch.load('../data/casp11/PSSMTesting.pt')

result_dir = "{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
    root='results',
    runner_name='run_amazon',
    date=datetime.now(),
)

os.makedirs(result_dir)
logfile_loc = "{}/{}.log".format(result_dir, 'output')
LOG = log.setup_custom_logger('runner', logfile_loc, 'Standard')
LOG.info('---------Listing all parameters-------')


casp_training = './data/casp11/training_90.pnet'
casp_validation = './data/casp11/validation.pnet'
casp_testing = './data/casp11/testing.pnet'

convert = ListToTorch()

args = parse_pnet(casp_testing,max_seq_len=1000)
AindTesting = args[1]
YobsTesting = args[5]
MSKTesting = args[8]
STesting = args[2]
AindTesting = convert(AindTesting)
YobsTesting = convert(YobsTesting)
MSKTesting = convert(MSKTesting)
STesting = convert(STesting)

args = parse_pnet(casp_validation,max_seq_len=1000)
AindVal = args[1]
YobsVal = args[5]
MSKVal = args[8]
SVal = args[2]
AindVal = convert(AindVal)
YobsVal = convert(YobsVal)
MSKVal = convert(MSKVal)
SVal = convert(SVal)

args = parse_pnet(casp_training,max_seq_len=1000)
Aind = args[1]
Yobs = args[5]
MSK = args[8]
S = args[2]
Aind = convert(Aind)
Yobs = convert(Yobs)
MSK = convert(MSK)
S = convert(S)

LOG.info('Number of data: {:}'.format(len(S)))
n_data_total = len(S)



def loss_tr_tuples(r1s,r2s, return_coords=False, coords_pred_std=None):
    '''
    Given two sets of 3D points of equal size. It computes the distance between these two sets of points, when allowing translation and rotation of the point clouds.
    # r1 -> Tensors of shape (b,3d,n)
    # r2 -> Tuples of length d, containing Tensors of shape (b,3,n)
    '''
    loss_tr = 0
    coords_pred = ()
    coords_target = ()
    for i,r2 in enumerate(r2s):
        r1 = r1s[:,3*i:3*i+3,:]

        mask = (r2 != 0).reshape(r2.shape)
        mask = (torch.sum(mask,dim=1) > 0).unsqueeze(1)
        batch_mask = torch.sum(mask,dim=(1,2)) > 10 # There needs to be at least 2 points in a protein for it to make sense, 1 point is nothing. 2 Might also be causing trouble so now we try 20
        mask = mask.repeat(1,3,1)

        r1 = r1[batch_mask,:,:]
        r2 = r2[batch_mask,:,:]
        mask = mask[batch_mask,:,:]



        #First we translate the two sets, by setting both their centroids to origin
        r1c = r1 - torch.sum(r1 * mask, dim=2, keepdim=True) / torch.sum(mask, dim=2, keepdim=True)
        r2c = r2 - torch.sum(r2 * mask, dim=2, keepdim=True) / torch.sum(mask, dim=2, keepdim=True)
        r1c = r1c * mask
        r2c = r2c * mask

        H = torch.bmm(r1c,r2c.transpose(1,2))
        U, S, V = torch.svd(H)

        d = torch.sign(torch.det(torch.bmm(V, U.transpose(1,2))))

        ones = torch.ones_like(d)
        a = torch.stack((ones, ones, d), dim=-1)
        tmp = torch.diag_embed(a)

        R = torch.bmm(V, torch.bmm(tmp, U.transpose(1,2)))

        r1cr = torch.bmm(R, r1c)
        if coords_pred_std is None:
            loss_tr += torch.mean(torch.norm(r1cr - r2c, dim=(1, 2)) ** 2 / torch.norm(r2c, dim=(1, 2)) ** 2) # TODO THIS IS WRONG?!
        else:
            r1_std = coords_pred_std[:, 3 * i:3 * i + 3, :]
            r1_std = r1_std[batch_mask, :, :]
            r1cr_std = torch.sqrt(torch.bmm(R**2, r1_std**2))

            dif = torch.abs(r1cr - r2c)
            m = dif > r1cr_std
            M = m * mask

            loss_tr += torch.mean(torch.norm((dif-r1cr_std)/(r1cr_std+1e-10)*M, dim=(1, 2)) ** 2 / torch.norm(r2c, dim=(1, 2)) ** 2)
        if return_coords:
            coords_pred += (r1cr.cpu().detach().numpy(),)
            coords_target += (r2c.cpu().detach().numpy(),)
    loss_tr / len(r2s)
    if return_coords:
        return loss_tr, coords_pred, coords_target
    else:
        return loss_tr


def getIterData(S, Aind, Yobs, MSK, i, device='cpu'):
    scale = 1e-3
    PSSM = S[i].t()
    n = PSSM.shape[1]
    M = MSK[i][:n]
    a = Aind[i]

    # X = Yobs[i][0, 0, :n, :n]
    X = Yobs[i].t()
    X = utils.linearInterp1D(X, M)
    X = torch.tensor(X)

    X = X - torch.mean(X, dim=1, keepdim=True)
    U, Lam, V = torch.svd(X)

    Coords = scale * torch.diag(Lam) @ V.t()
    Coords = Coords.type('torch.FloatTensor')

    PSSM = PSSM.type(torch.float32)
    PSSM = augmentPSSM(PSSM, 0.05)

    A = torch.zeros(20, n)
    A[a, torch.arange(0, n)] = 1.0
    Seq = torch.cat((PSSM, A))
    # Seq = torch.cat((A,-A))

    Seq = Seq.to(device=device, non_blocking=True)

    Coords = Coords.to(device=device, non_blocking=True)
    M = M.type('torch.FloatTensor')
    M = M.to(device=device, non_blocking=True)

    return Seq, Coords, M


def augmentPSSM(Z, sig=1):
    Uz, Laz, Vz = torch.svd(Z)
    n = len(Laz)
    r = torch.rand(n) * Laz.max() * sig
    Zout = Uz @ torch.diag((1 + r) * Laz) @ Vz.t()
    Zout = torch.relu(Zout)
    Zout = Zout / (torch.sum(Zout, dim=0, keepdim=True) + 0.001)
    return Zout

Z,C,M = getIterData(S,Aind,Yobs,MSK,10)
# plt.imshow(Z[:,:50])
Z.device

Seq, Coords, M = getIterData(S,Aind,Yobs,MSK,11)
LOG.info(Seq.shape)
LOG.info(Coords.shape)
LOG.info(M.shape)

MM = torch.ger(M,M)
DMt = utils.getDistMat(Coords)
dm  = DMt.max()
sig = 0.3
D   = torch.exp(-DMt/(dm*sig))
# plt.subplot(1,2,1)
# plt.imshow(MM*DMt)
# plt.colorbar()
# plt.subplot(1,2,2)
# plt.imshow(MM*D)
# plt.colorbar()

nstart  = 3
nopen   = 128
nhid    = 256
nclose  = 40
nlayers = 50
h       = 1/nlayers
Arch = [nstart, nopen, nhid, nclose, nlayers]

#model = networks.graphNN(Arch)
model = networks.hyperNet(Arch,h)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
LOG.info('Number of parameters {:}'.format(total_params))

lrO = 1e-2 #1e-4
lrC = 1e-2 #1e-3
lrN = 1e-2 #1e-4
lrB = 1e-2 #1e-4

optimizer = optim.Adam([{'params': model.Kopen, 'lr': lrO},
                        {'params': model.Kclose, 'lr': lrC},
                        {'params': model.W, 'lr': lrN},
                        {'params': model.Bias, 'lr': lrB}])

#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2)

alossBest = 1e6
ndata = len(S)
epochs = 30
sig   = 0.3
ndata = n_data_total
bestModel = model
hist = torch.zeros(epochs)
max_iter = epochs*ndata
LOG.info('         Design       Coords      Reg           gradW       gradKo        gradKc       gradB')
for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    amis = 0.0
    amisb = 0.0
    for i in range(ndata):

        Z, Coords, M = getIterData(S, Aind, Yobs, MSK, i, device=device)
        M = torch.ger(M, M)

        optimizer.zero_grad()
        # From Coords to Seq
        Zout, Zold = model(Coords)
        PSSMpred = F.softshrink(Zout[:20, :].abs(), Zout.abs().mean().item() / 5)
        misfit = utils.kl_div(PSSMpred, Z[:20, :], weight=True)

        # From Seq to Coord
        Cout, CoutOld = model.backwardProp(Z)
        d = torch.sqrt(torch.sum((Coords[:, 1:] - Coords[:, :-1]) ** 2, dim=0)).mean()
        Cout = utils.distConstraint(Cout, d)
        CoutOld = utils.distConstraint(CoutOld, d)

        DM = utils.getDistMat(Cout)
        DMt = utils.getDistMat(Coords)
        dm = DMt.max()
        D = torch.exp(-DM / (dm * sig))
        Dt = torch.exp(-DMt / (dm * sig))
        # misfitBackward = torch.norm(M * Dt - M * D) ** 2 / torch.norm(M * Dt) ** 2
        # W = 1/(DMt+1e-4*torch.ones(DMt.shape[0], device=device))
        misfitBackward_dist = torch.norm((M * Dt - M * D)) ** 2 / torch.norm((M * Dt)) ** 2
        ite = j*ndata+i
        w = ite / max_iter
        misfitBackward_coord = loss_tr_tuples(Cout[None,:,:], (Coords[None,:,:],))
        misfitBackward = (1 - w) / 2 * misfitBackward_dist + (w + 1) / 2 * misfitBackward_coord

        R = model.NNreg()
        C0 = torch.norm(Cout - CoutOld) ** 2 / torch.numel(Z)
        Z0 = torch.norm(Zout - Zold) ** 2 / torch.numel(Z)
        loss = misfit + misfitBackward + R + C0 + Z0

        loss.backward(retain_graph=True)

        aloss += loss.detach()
        amis += misfit.detach().item()
        amisb += misfitBackward.detach().item()

        optimizer.step()
        # scheduler.step()
        nprnt = 1000
        if (i + 1) % nprnt == 0:
            amis = amis / nprnt
            amisb = amisb / nprnt
            LOG.info("{:2d}.{:1d}  {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e}".format(j, i, amis, amisb, R.item(),model.W.grad.norm().item(), model.Kopen.grad.norm().item(), model.Kclose.grad.norm().item(),model.Bias.grad.norm().item()))
            amis = 0.0
            amisb = 0.0
    if aloss < alossBest:
        alossBest = aloss
        bestModel = model

    # Validation on 0-th data
    with torch.no_grad():
        misVal = 0
        misbVal = 0
        AQdis = 0
        # nVal    = len(SVal)
        nVal = len(STesting)
        for jj in range(nVal):
            # Z, Coords, M = getIterData(SVal, AindVal, YobsVal, MSKVal, jj,device=device)
            Z, Coords, M = getIterData(STesting, AindTesting, YobsTesting, MSKTesting, jj, device=device)
            M = torch.ger(M, M)
            Zout, Zold = model(Coords)
            PSSMpred = F.softshrink(Zout[:20, :].abs(), Zout.abs().mean().item() / 5)
            misfit = utils.kl_div(PSSMpred, Z[:20, :], weight=True)

            misVal += misfit
            # From Seq to Coord
            Cout, CoutOld = model.backwardProp(Z)
            d = torch.sqrt(torch.sum((Coords[:, 1:] - Coords[:, :-1]) ** 2, dim=0)).mean()
            Cout = utils.distConstraint(Cout, d)
            CoutOld = utils.distConstraint(CoutOld, d)

            DM = utils.getDistMat(Cout)
            DMt = utils.getDistMat(Coords)
            dm = DMt.max()
            D = torch.exp(-DM / (dm * sig))
            Dt = torch.exp(-DMt / (dm * sig))
            # misfitBackward = torch.norm(M*DMt-M*DM)**2/torch.norm(M*DMt)**2
            # W = 1/(DMt+1e-4*torch.ones(DMt.shape[0], device=device))
            misfitBackward = torch.norm((M * Dt - M * D)) ** 2 / torch.norm((M * Dt)) ** 2

            misbVal += misfitBackward
            AQdis += torch.norm(M * (DM - DMt)) / M.shape[0] / (M.shape[1] - 1)

            # GDT_score

        LOG.info("{:2d}       {:10.3e}   {:10.3e}   {:10.3e}" % (j, misVal / nVal, misbVal / nVal, AQdis / nVal))
        LOG.info('===============================================')

    hist[j] = (aloss).item() / (ndata)

    with torch.no_grad():
        misVal = 0
        misbVal = 0
        AQdis = 0
        TrueD = []
        RecD = []
        MM = []
        nVal = len(STesting)
        for jj in range(nVal):
            Z, Coords, M = getIterData(STesting, AindTesting, YobsTesting, MSKTesting, jj, device=device)
            M = torch.ger(M, M)
            Zout, Zold = model(Coords)
            PSSMpred = F.softshrink(Zout[:20, :].abs(), Zout.abs().mean().item() / 5)
            misfit = utils.kl_div(PSSMpred, Z[:20, :], weight=True)
            misVal += misfit
            # From Seq to Coord
            Cout, CoutOld = model.backwardProp(Z)
            d = torch.sqrt(torch.sum((Coords[:, 1:] - Coords[:, :-1]) ** 2, dim=0)).mean()
            Cout = utils.distConstraint(Cout, d)
            CoutOld = utils.distConstraint(CoutOld, d)

            DM = utils.getDistMat(Cout)
            DMt = utils.getDistMat(Coords)
            TrueD.append(DM.detach().cpu())
            RecD.append(DMt.detach().cpu())
            MM.append(M.detach().cpu())

            dm = DMt.max()
            D = torch.exp(-DM / (dm * sig))
            Dt = torch.exp(-DMt / (dm * sig))
            misfitBackward = torch.norm(M * Dt - M * D) ** 2 / torch.norm(M * Dt) ** 2
            misbVal += misfitBackward
            aqi = torch.norm(M * (DM - DMt)) / M.shape[0] / (M.shape[1] - 1)
            AQdis += aqi
            # print(aqi.item())

        LOG.info("{:2d}       {:10.3e}   {:10.3e}   {:10.3e}" % (j, misVal / nVal, misbVal / nVal, AQdis / nVal))
        LOG.info('===============================================')

torch.save(TrueD, 'TrueTestingDistanceWithConst.pt')
torch.save(RecD, 'RecTestingDistanceWithConst.pt')
torch.save(MM, 'MaskTestingDistanceWithConst.pt')

# plt.subplot(1,2,1)
# plt.imshow(Z[:20,:40].detach().cpu())
# plt.subplot(1,2,2)
# plt.imshow(F.softmax(F.softshrink(Zout[:20,:40]),dim=0).detach().cpu())
#
# jj = 9
# plt.subplot(1,2,1)
# plt.imshow(MM[jj]*TrueD[jj])
# plt.subplot(1,2,2)
# plt.imshow(MM[jj]*RecD[jj])

torch.norm((TrueD[jj]-RecD[jj]))/TrueD[jj].shape[0]/(TrueD[jj].shape[1]-1)

# plt.imshow(Dt.cpu())
# plt.colorbar()


# Coords.device
# tt = torch.zeros(3,3,device=Coords.device)

# SVal     = torch.load('../data/casp11/PSSMVal.pt')
