import os, sys
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


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load training data
Aind = torch.load('../../../data/casp11/AminoAcidIdx.pt')
Yobs = torch.load('../../../data/casp11/RCalpha.pt')
MSK  = torch.load('../../../data/casp11/Masks.pt')
S     = torch.load('../../../data/casp11/PSSM.pt')
# load validation data
AindVal = torch.load('../../../data/casp11/AminoAcidIdxVal.pt')
YobsVal = torch.load('../../../data/casp11/RCalphaVal.pt')
MSKVal  = torch.load('../../../data/casp11/MasksVal.pt')
SVal     = torch.load('../../../data/casp11/PSSMVal.pt')

# load Testing data
AindTesting = torch.load('../../../data/casp11/AminoAcidIdxTesting.pt')
YobsTesting = torch.load('../../../data/casp11/RCalphaTesting.pt')
MSKTesting  = torch.load('../../../data/casp11/MasksTesting.pt')
STesting     = torch.load('../../../data/casp11/PSSMTesting.pt')

print('Number of data: ', len(S))
n_data_total = len(S)


def getIterData(S, Aind, Yobs, MSK, i, device='cpu', pad=0):
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
    # PSSM = augmentPSSM(PSSM, 0.01)

    A = torch.zeros(20, n)
    A[a, torch.arange(0, n)] = 1.0
    Seq = torch.cat((PSSM, A))

    if pad > 0:
        L = Coords.shape[1]
        k = 2 ** torch.tensor(L, dtype=torch.float64).log2().ceil().int()
        k = k.item()
        CoordsPad = torch.zeros(3, k)
        CoordsPad[:, :Coords.shape[1]] = Coords
        SeqPad = torch.zeros(Seq.shape[0], k)
        SeqPad[:, :Seq.shape[1]] = Seq
        Mpad = torch.zeros(k)
        MM = torch.zeros(k)
        Mpad[:M.shape[0]] = torch.ones(M.shape[0], device=M.device)
        M = Mpad
        MM[:M.shape[0]] = M
        M = MM
        Seq = SeqPad
        Coords = CoordsPad

    Seq = Seq.to(device=device, non_blocking=True)
    Coords = Coords.to(device=device, non_blocking=True)
    M = M.type('torch.FloatTensor')
    M = M.to(device=device, non_blocking=True)
    Mpad = Mpad.type('torch.FloatTensor')
    Mpad = Mpad.to(device=device, non_blocking=True)

    return Seq, Coords, M, Mpad


# Unet Architecture
nLevels  = 3
nin      = 40
nsmooth  = 2
nopen    = 4 #64
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
epochs = 3
sig   = 0.3
ndata = 1000 #n_data_total
bestModel = model
hist = torch.zeros(epochs)

print('         Design       Coords              gradKo        gradKc')
for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    amis = 0.0
    amisb = 0.0
    for i in range(ndata):

        Z, Coords, M, Mpad = getIterData(S, Aind, Yobs, MSK, i, device=device, pad=1)
        kk = i
        flag = False
        while torch.sum(M) < 0.75 * M.numel():
            if kk + 1 < ndata:
                Z, Coords, M, Mpad = getIterData(S, Aind, Yobs, MSK, kk + 1, device=device, pad=1)
                kk += 1
            else:
                flag = True
        if flag:
            break

        Z = Z.unsqueeze(0)
        Coords = Coords.unsqueeze(0)
        M = M.unsqueeze(0).unsqueeze(0)
        Mpad = Mpad.unsqueeze(0).unsqueeze(0)

        optimizer.zero_grad()
        # From Coords to Seq
        Cout, CoutOld = model(Z, Mpad)
        Cout = utils.distConstraint(Cout, dc=0.379, M=M).unsqueeze(0)
        CoutOld = utils.distConstraint(CoutOld, dc=0.379, M=M).unsqueeze(0)

        Zout, Zold = model.backProp(Coords, Mpad)

        PSSMpred = F.softshrink(Zout[0, :20, :].abs(), Zout.abs().mean().item() / 5)
        misfit = utils.kl_div(PSSMpred, Z[0, :20, :], weight=True)

        DM = utils.getDistMat(Cout.squeeze(0))
        DMt = utils.getDistMat(Coords.squeeze(0))
        dm = DMt.max()
        D = torch.exp(-DM / (dm * sig))
        Dt = torch.exp(-DMt / (dm * sig))
        MM = torch.ger(M.squeeze(), M.squeeze())
        # misfitBackward = torch.norm(MM * Dt - MM * D) ** 2 / torch.norm(MM * Dt) ** 2
        misfitBackward, _, _ = utils.coord_loss(Cout, Coords, M)

        # R = model.NNreg()
        C0 = torch.norm(Cout - CoutOld) ** 2 / torch.numel(Z)
        Z0 = torch.norm(Zout - Zold) ** 2 / torch.numel(Z)
        loss = misfit + misfitBackward + C0 + Z0

        # loss.backward(retain_graph=True)
        loss.backward()

        aloss += loss.detach()
        amis += misfit.detach().item()
        amisb += misfitBackward.detach().item()

        gKopen = model.Kopen.grad.norm().item()
        gKclose = model.Kclose.grad.norm().item()

        optimizer.step()
        # scheduler.step()
        nprnt = 1000
        if (i + 1) % nprnt == 0:
            amis = amis / nprnt
            amisb = amisb / nprnt
            print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E " %
                  (j, i, amis, amisb, gKopen, gKclose))
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
            Z, Coords, M, Mpad = getIterData(STesting, AindTesting, YobsTesting, MSKTesting, jj, device=device, pad=1)
            Coords = Coords.unsqueeze(0)
            Z = Z.unsqueeze(0)
            M = M.unsqueeze(0).unsqueeze(0)
            Mpad = Mpad.unsqueeze(0).unsqueeze(0)

            optimizer.zero_grad()
            # From Coords to Seq
            Cout, CoutOld = model(Z, Mpad)
            Cout = utils.distConstraint(Cout, dc=0.379, M=M).unsqueeze(0)
            CoutOld = utils.distConstraint(CoutOld, dc=0.379, M=M).unsqueeze(0)

            Zout, ZOld = model.backProp(Coords, Mpad)

            PSSMpred = F.softshrink(Zout[0, :20, :].abs(), Zout.abs().mean().item() / 5)
            misfit = utils.kl_div(PSSMpred, Z[0, :20, :], weight=True)
            misVal += misfit

            MM = torch.ger(M.squeeze(), M.squeeze())
            DM = utils.getDistMat(Cout.squeeze(0))
            DMt = utils.getDistMat(Coords.squeeze(0))
            dm = DMt.max()
            D = torch.exp(-DM / (dm * sig))
            Dt = torch.exp(-DMt / (dm * sig))
            # misfitBackward = torch.norm(MM * Dt - MM * D) ** 2 / torch.norm(MM * Dt) ** 2
            misfitBackward, _, _ = utils.coord_loss(Cout, Coords, M)

            misbVal += misfitBackward
            AQdis += torch.norm(MM * (DM - DMt)) / torch.sum(MM > 0)

        print("%2d       %10.3E   %10.3E   %10.3E" % (j, misVal / nVal, misbVal / nVal, AQdis / nVal))
        print('===============================================')

    hist[j] = (aloss).item() / (ndata)


