import os, sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from src import networks
from src import pnetProcess
from src import utils
import matplotlib.pyplot as plt
import torch.optim as optim
#from src import networks
from src import graphUnetworks as gunts
from src.dataloader_pnet import ListToTorch, parse_pnet
from srcOld import log

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# load training data
# Aind = torch.load('./data/casp11/Aind.pt')
# Yobs = torch.load('./data/casp11/Yobs.pt')
# MSK  = torch.load('./data/casp11/MSK.pt')
# S     = torch.load('./data/casp11/S.pt')
# # load validation data
# AindVal = torch.load('./data/casp11/AindVal.pt')
# YobsVal = torch.load('./data/casp11/YobsVal.pt')
# MSKVal  = torch.load('./data/casp11/MSKVal.pt')
# SVal     = torch.load('./data/casp11/SVal.pt')
#
# # load Testing data
# AindTesting = torch.load('./data/casp11/AindTesting.pt')
# YobsTesting = torch.load('./data/casp11/YobsTesting.pt')
# MSKTesting  = torch.load('./data/casp11/MSKTesting.pt')
# STesting     = torch.load('./data/casp11/STesting.pt')
#



Aind = torch.load('./../data/casp11_protein_design/Aind.pt')
Yobs = torch.load('./../data/casp11_protein_design/Yobs.pt')
MSK  = torch.load('./../data/casp11_protein_design/MSK.pt')
S     = torch.load('./../data/casp11_protein_design/S.pt')
# load validation data
AindVal = torch.load('./../data/casp11_protein_design/AindVal.pt')
YobsVal = torch.load('./../data/casp11_protein_design/YobsVal.pt')
MSKVal  = torch.load('./../data/casp11_protein_design/MSKVal.pt')
SVal     = torch.load('./../data/casp11_protein_design/SVal.pt')

# load Testing data
AindTesting = torch.load('./../data/casp11_protein_design/AindTesting.pt')
YobsTesting = torch.load('./../data/casp11_protein_design/YobsTesting.pt')
MSKTesting  = torch.load('./../data/casp11_protein_design/MSKTesting.pt')
STesting     = torch.load('./../data/casp11_protein_design/STesting.pt')


result_dir = "{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
    root='results',
    runner_name='run_amazon',
    date=datetime.now(),
)
enable_coord_loss = True
os.makedirs(result_dir)
logfile_loc = "{}/{}.log".format(result_dir, 'output')
LOG = log.setup_custom_logger('runner', logfile_loc, 'Standard')
LOG.info('---------Listing all parameters-------')
LOG.info("Coordinate loss = {:}".format(enable_coord_loss))
#
#
# casp_training = './data/casp11/training_90.pnet'
# casp_validation = './data/casp11/validation.pnet'
# casp_testing = './data/casp11/testing.pnet'
#
# convert = ListToTorch()
#
# args = parse_pnet(casp_testing,max_seq_len=1000)
# AindTesting = args[1]
# YobsTesting = args[5]
# MSKTesting = args[8]
# STesting = args[2]
# AindTesting = convert(AindTesting)
# YobsTesting = convert(YobsTesting)
# MSKTesting = convert(MSKTesting)
# STesting = convert(STesting)
#
# args = parse_pnet(casp_validation,max_seq_len=1000)
# AindVal = args[1]
# YobsVal = args[5]
# MSKVal = args[8]
# SVal = args[2]
# AindVal = convert(AindVal)
# YobsVal = convert(YobsVal)
# MSKVal = convert(MSKVal)
# SVal = convert(SVal)
#
# args = parse_pnet(casp_training,max_seq_len=1000)
# Aind = args[1]
# Yobs = args[5]
# MSK = args[8]
# S = args[2]
# Aind = convert(Aind)
# Yobs = convert(Yobs)
# MSK = convert(MSK)
# S = convert(S)
#
# torch.save(Aind,'./data/casp11/Aind.pt')
# torch.save(Yobs,'./data/casp11/Yobs.pt')
# torch.save(MSK,'./data/casp11/MSK.pt')
# torch.save(S,'./data/casp11/S.pt')
# torch.save(AindVal,'./data/casp11/AindVal.pt')
# torch.save(YobsVal,'./data/casp11/YobsVal.pt')
# torch.save(MSKVal,'./data/casp11/MSKVal.pt')
# torch.save(SVal,'./data/casp11/SVal.pt')
# torch.save(AindTesting,'./data/casp11/AindTesting.pt')
# torch.save(YobsTesting,'./data/casp11/YobsTesting.pt')
# torch.save(MSKTesting,'./data/casp11/MSKTesting.pt')
# torch.save(STesting,'./data/casp11/STesting.pt')

print('Number of data: ', len(S))
n_data_total = len(S)


class Loss_reg(torch.nn.Module):
    def __init__(self,d_mean,d_std,device='cpu'):
        super(Loss_reg,self).__init__()
        A = torch.from_numpy(d_mean)
        AA = A + A.T
        AA[torch.arange(AA.shape[0]),torch.arange(AA.shape[0])] /= 2
        self.d_mean = AA.to(device=device, dtype=torch.float32)

        A = torch.from_numpy(d_std)
        AA = A + A.T
        AA[torch.arange(AA.shape[0]),torch.arange(AA.shape[0])] /= 2
        self.d_std = AA.to(device=device, dtype=torch.float32)
        return

    def forward(self, seq, coord_pred, mask_padding):
        #We only want places where the target is larger than zero (remember this is for distances)

        d = torch.squeeze(torch.norm(coord_pred[:, :, 1:] - coord_pred[:, :, :-1], 2, dim=1))
        d_upper = self.d_mean[seq[:,1:],seq[:,:-1]] + 3 * self.d_std[seq[:,1:],seq[:,:-1]]
        d_lower = self.d_mean[seq[:,1:],seq[:,:-1]] - 3 * self.d_std[seq[:,1:],seq[:,:-1]]

        m1 = d < d_lower
        m2 = d > d_upper

        M = (m1 + m2) * mask_padding[:,1:]

        df_u = torch.abs(d - d_upper)
        df_l = torch.abs(d - d_lower)
        df_all = torch.stack((df_u,df_l))
        df = torch.min(df_all,dim=0)[0]

        # loss = torch.mean(torch.norm(df*M,1,dim=1))
        loss = torch.mean(torch.norm(df*M,1,dim=1)/torch.sum(mask_padding[:,1:],dim=1))


        # loss = torch.mean((torch.norm((d-d_reg)*mask_padding[:,1:],1,dim=1)))

        return loss


class Loss_reg_min_separation(torch.nn.Module):
    def __init__(self,d_mean=0.3432,d_std=0.0391):
        super(Loss_reg_min_separation,self).__init__()
        self.d = d_mean - 3 * d_std
        return

    def forward(self, dists, mask_padding):
        #We only want places where the target is larger than zero (remember this is for distances)
        M = mask_padding[:,:, None].float() @ mask_padding[:,None,:].float()
        dists = dists[0]
        mask_diag = dists > 0
        mask_relevant = dists < self.d
        M2 = M * mask_diag * mask_relevant
        # loss = torch.mean(torch.norm((dists-self.d)*M2,1,dim=(1,2)))
        loss = torch.mean(torch.norm((dists-self.d)*M2,1,dim=(1,2))/torch.sum(M,dim=(1,2)))

        # d = torch.squeeze(torch.norm(coord_pred[:, :, 1:] - coord_pred[:, :, :-1], 2, dim=1))
        # d_reg = self.d_mean[seq[:,1:],seq[:,:-1]]
        # loss = torch.mean((torch.norm((d-d_reg)*mask_padding[:,1:],1,dim=1)))

        return loss



data = np.load('./data/binding_distances.npz')
d_mean = data['d_mean']
d_std = data['d_std']
loss_reg_fnc = Loss_reg(d_mean,d_std,device=device)
loss_reg_min_sep_fnc = Loss_reg_min_separation()




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


def getIterData(S, Aind, Yobs, MSK, i, device='cpu',pad=0):
    scale = 1e-3
    PSSM = S[i].t()
    n = PSSM.shape[1]
    M = MSK[i][:n]
    a = Aind[i]

    X = Yobs[i].t()
    X = utils.linearInterp1D(X, M)
    X = torch.tensor(X)

    X = X - torch.mean(X, dim=1, keepdim=True)
    U, Lam, V = torch.svd(X)

    Coords = scale * torch.diag(Lam) @ V.t()
    Coords = Coords.type('torch.FloatTensor')

    PSSM = PSSM.type(torch.float32)

    A = torch.zeros(20, n)
    A[a, torch.arange(0, n)] = 1.0
    Seq = torch.cat((PSSM, A))
    Seq = Seq.to(device=device, non_blocking=True)

    Coords = Coords.to(device=device, non_blocking=True)
    M = M.type('torch.FloatTensor')
    M = M.to(device=device, non_blocking=True)

    if pad > 0:
        L = Coords.shape[1]
        k = 2**torch.tensor(L, dtype=torch.float64).log2().ceil().int()
        k = k.item()
        CoordsPad = torch.zeros(3,k,device=device)
        CoordsPad[:,:Coords.shape[1]] = Coords
        SeqPad    = torch.zeros(Seq.shape[0],k,device=device)
        SeqPad[:,:Seq.shape[1]] = Seq
        Mpad      = torch.zeros(k,device=device)
        Mpad[:M.shape[0]] = M
        M = Mpad
        Seq = SeqPad
        Coords = CoordsPad

    return Seq, Coords, M

# Unet Architecture
nLevels  = 3
nin      = 40
nsmooth  = 2
nopen    = 64
nLayers  = 18
nout     = 3
h        = 0.1

model = gunts.stackedGraphUnet(nLevels,nsmooth,nin,nopen,nLayers,nout,h)
model.to(device)


lrO = 1e-5
lrC = 1e-5
lrU = 1e-4

optimizer = optim.Adam([{'params': model.Kopen, 'lr': lrO},
                        {'params': model.Kclose, 'lr': lrC},
                        {'params': model.Unets.parameters(), 'lr': lrU}])

alossBest = 1e6
epochs = 30
sig = 0.3
ndata = n_data_total
# ndata = 1
bestModel = model
hist = torch.zeros(epochs)
max_iter = epochs*ndata
t0 = time.time()
print('         Design       Coords      gradKo        gradKc')
for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    amis = 0.0
    amisb = 0.0
    loss_regs = 0.0
    for i in range(ndata):

        Z, Coords, M = getIterData(S, Aind, Yobs, MSK, i, device=device, pad=1)
        Z = Z.unsqueeze(0)
        Coords = Coords.unsqueeze(0)
        M = M.unsqueeze(0).unsqueeze(0)

        optimizer.zero_grad()
        # From Coords to Seq
        Cout, CoutOld = model(Z, M)
        Zout, Zold    = model.backProp(Coords,M)

        PSSMpred = F.softshrink(Zout[0,:20, :].abs(), Zout.abs().mean().item() / 5)
        misfit = utils.kl_div(PSSMpred, Z[0,:20, :], weight=True)

        #d = torch.sqrt(torch.sum((Coords[0,:, 1:] - Coords[0,:, :-1]) ** 2, dim=0))
        #Cout    = utils.distConstraint(Cout.squeeze(0), d).unsqueeze(0)
        #CoutOld = utils.distConstraint(CoutOld.squeeze(0), d).unsqueeze(0)

        MM = torch.ger(M.squeeze(0).squeeze(0), M.squeeze(0).squeeze(0))
        DM = utils.getDistMat(Cout.squeeze(0))
        DMt = utils.getDistMat(Coords.squeeze(0))
        dm = DMt.max()
        D = torch.exp(-DM / (dm * sig))
        Dt = torch.exp(-DMt / (dm * sig))
        misfitBackward_dist = torch.norm((MM * Dt - MM * D)) ** 2 / torch.norm((MM * Dt)) ** 2
        ite = j * ndata + i
        if enable_coord_loss:
            w = ite / max_iter
            misfitBackward_coord = loss_tr_tuples(Cout[:,:], (Coords[:,:],))
            misfitBackward = (1 - w) / 2 * misfitBackward_dist + (w + 1) / 2 * misfitBackward_coord
        else:
            misfitBackward = misfitBackward_dist

        seq = torch.argmax(Z[:, 20:, :], dim=1)
        loss_reg = 10 * loss_reg_fnc(seq, Cout, M[0,:,:])
        loss_reg_min_sep = 10 * loss_reg_min_sep_fnc(DM, M[0,:,:])

        #R = model.NNreg()
        C0 = torch.norm(Cout - CoutOld) ** 2 / torch.numel(Z)
        Z0 = torch.norm(Zout - Zold) ** 2 / torch.numel(Z)
        loss = misfit + misfitBackward + C0 + Z0 + loss_reg + loss_reg_min_sep

        loss.backward(retain_graph=True)

        aloss += loss.detach()
        amis += misfit.detach().item()
        amisb += misfitBackward.detach().item()
        loss_regs += loss_reg.detach().item() + loss_reg_min_sep.detach().item()

        optimizer.step()
        # scheduler.step()
        nprnt = 100
        if (i + 1) % nprnt == 0:
            amis = amis / nprnt
            amisb = amisb / nprnt
            loss_regs = loss_regs / nprnt
            t1 = time.time()
            LOG.info("{:2d}.{:1d}  {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:2.2f}h, ETA={:2.2f}h".format(j, i, amis, amisb, model.Kopen.grad.norm().item(), model.Kclose.grad.norm().item(),loss_regs,(t1-t0)/3600,(max_iter-ite+1)/(ite+1)*(t1-t0)/3600))
            amis = 0.0
            amisb = 0.0
            loss_regs = 0.0
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
            Z, Coords, M = getIterData(STesting, AindTesting, YobsTesting, MSKTesting, jj, device=device, pad=1)
            Coords = Coords.unsqueeze(0)
            Z = Z.unsqueeze(0)
            M = M.unsqueeze(0).unsqueeze(0)

            optimizer.zero_grad()
            # From Coords to Seq
            Cout, CoutOld = model(Z, M)
            Zout, ZOld = model.backProp(Coords, M)

            PSSMpred = F.softshrink(Zout[0,:20, :].abs(), Zout.abs().mean().item() / 5)
            misfit = utils.kl_div(PSSMpred, Z[0,:20, :], weight=True)
            misVal += misfit

            #d = torch.sqrt(torch.sum((Coords[0,:, 1:] - Coords[0,:, :-1]) ** 2, dim=0)).mean()
            #Cout = utils.distConstraint(Cout.squeeze(0), d).unsqueeze(0)
            #CoutOld = utils.distConstraint(CoutOld.squeeze(0), d).unsqueeze(0)

            MM = torch.ger(M.squeeze(0).squeeze(0), M.squeeze(0).squeeze(0))
            DM = utils.getDistMat(Cout.squeeze(0))
            DMt = utils.getDistMat(Coords.squeeze(0))
            dm = DMt.max()
            D = torch.exp(-DM / (dm * sig))
            Dt = torch.exp(-DMt / (dm * sig))
            misfitBackward = torch.norm(MM * Dt - MM * D) ** 2 / torch.norm(MM * Dt) ** 2


            misbVal += misfitBackward
            AQdis += torch.norm(MM * (DM - DMt)) / torch.sum(MM>0)

        LOG.info("{:}       {:10.3e}   {:10.3e}   {:10.3e}".format(j, misVal / nVal, misbVal / nVal, AQdis / nVal))
        LOG.info('===============================================')

    hist[j] = (aloss).item() / (ndata)