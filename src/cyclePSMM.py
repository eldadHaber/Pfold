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
from src import networks

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#X    = torch.load('../../../data/casp12Testing/Xtest.pt')
#Yobs = torch.load('../../../data/casp12Testing/Coordtest.pt')
#MSK  = torch.load('../../../data/casp12Testing/Masktest.pt')
#S     = torch.load('../../../data/casp12Testing/Seqtest.pt')

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

print('Number of data: ', len(S))
n_data_total = len(S)


def getIterData(S, Aind, i, device='cpu'):
    scale = 1e-3
    PSSM = S[i].t()
    n = PSSM.shape[1]
    a = Aind[i]

    PSSM = PSSM.type(torch.float32)
    PSSM = augmentPSSM(PSSM, 0.3)
    PSSM = PSSM.to(device=device, non_blocking=True)

    Seq = torch.zeros(20, n)
    Seq[a, torch.arange(0, n)] = 1.0
    Seq = Seq.to(device=device, non_blocking=True)

    return Seq, PSSM


def augmentPSSM(Z, sig=1):
    Uz, Laz, Vz = torch.svd(Z)
    n = len(Laz)
    r = torch.rand(n) * Laz.max() * sig
    Zout = Uz @ torch.diag((1 + r) * Laz) @ Vz.t()
    Zout = torch.relu(Zout)
    Zout = Zout / (torch.sum(Zout, dim=0, keepdim=True) + 0.001)
    return Zout

nstart  = 20
nopen   = 64
nhid    = 128
nclose  = 20
nlayers = 50
h       = 1/nlayers
Arch = [nstart, nopen, nhid, nclose, nlayers]

#model = networks.graphNN(Arch)
model = networks.hyperNet(Arch,h)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ',total_params)


lrO = 1e-4
lrC = 1e-3
lrN = 1e-4
lrB = 1e-4

optimizer = optim.Adam([{'params': model.Kopen, 'lr': lrO},
                        {'params': model.Kclose, 'lr': lrC},
                        {'params': model.W, 'lr': lrN},
                        {'params': model.Bias, 'lr': lrB}])

alossBest = 1e6
ndata = len(S)
epochs = 1000
sig   = 0.3
ndata = 1000 #n_data_total
bestModel = model
hist = torch.zeros(epochs)

print('         Design       Coords      Reg           gradW       gradKo        gradKc       gradB')
for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    amis = 0.0
    amisb = 0.0
    for i in range(ndata):

        Seq, PSSM = getIterData(S, Aind, i, device=device)

        optimizer.zero_grad()
        # From Coords to Seq
        PSSMout, PSSMold = model(Seq)
        Seqout,  SeqOld = model.backwardProp(PSSM)

        onehotZ = Aind[i].to(device)
        wloss = torch.zeros(20).to(device)
        for kk in range(20):
            wloss[kk] = torch.sum(onehotZ == kk)
        wloss = 1.0 / (wloss + 1.0)

        misfitFor  = F.kl_div(PSSMout.t().unsqueeze(0), PSSM.t().unsqueeze(0))
        misfitBack = F.cross_entropy(Seqout.t(), onehotZ, wloss)
        # From Seq to Coord

        R = model.NNreg()
        C0 = torch.norm(PSSMout - PSSMold) ** 2 / torch.numel(PSSM)
        Z0 = torch.norm(Seqout - SeqOld) ** 2 / torch.numel(PSSM)
        loss = misfitFor + misfitBackward + R + C0 + Z0

        loss.backward(retain_graph=True)

        aloss += loss.detach()
        amis += misfit.detach().item()
        amisb += misfitBackward.detach().item()

        optimizer.step()
        nprnt = 10
        if (i + 1) % nprnt == 0:
            amis = amis / nprnt
            amisb = amisb / nprnt
            print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E" %
                  (j, i, amis, amisb, R.item(),
                   model.W.grad.norm().item(), model.Kopen.grad.norm().item(), model.Kclose.grad.norm().item(),
                   model.Bias.grad.norm().item()))
            amis = 0.0
            amisb = 0.0
    if aloss < alossBest:
        alossBest = aloss
        bestModel = model

    # Validation on 0-th data
    with torch.no_grad():
        misVal = 0
        misbVal = 0
        nVal = 4 #len(SVal)
        for jj in range(nVal):
            Z, Coords, M = getIterData(SVal, AindVal, YobsVal, MSKVal, jj, device=device)
            M = torch.ger(M, M)
            Zout, Zold = model(Coords)
            #onehotZ = torch.argmax(Z[20:, :], dim=0)
            onehotZ = AindVal[jj].to(device)
            wloss = torch.zeros(20).to(device)
            for kk in range(20):
                wloss[kk] = torch.sum(onehotZ == kk)
            wloss = 1.0 / (wloss + 1.0)
            misfit = F.cross_entropy(Zout[20:, :].t(), onehotZ, wloss)
            misVal += misfit
            # From Seq to Coord
            Cout, CoutOld = model.backwardProp(Z)
            DM = utils.getDistMat(Cout)
            DMt = utils.getDistMat(Coords)
            dm = DMt.max()
            D = torch.exp(-DM / (dm * sig))
            Dt = torch.exp(-DMt / (dm * sig))
            misfitBackward = torch.norm(M * Dt - M * D) ** 2 / torch.norm(M * Dt) ** 2
            misbVal += misfitBackward

        print("%2d       %10.3E   %10.3E" % (j, misVal / nVal, misbVal / nVal))
        print('===============================================')

    hist[j] = (aloss).item() / (ndata)
