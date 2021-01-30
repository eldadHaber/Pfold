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



nstart  = 3
nopen   = 128
nhid    = 256
nclose  = 40
nlayers = 6
h       = 1/nlayers
Arch = [nstart, nopen, nhid, nclose, nlayers]

model = networks.hyperNet(Arch,h)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ',total_params)


lrO = 1e-2 #1e-5
lrC = 1e-2 #1e-5
lrN = 1e-2 # 1e-4
lrB = 1e-2 #1e-4

optimizer = optim.Adam([{'params': model.Kopen, 'lr': lrO},
                        {'params': model.Kclose, 'lr': lrC},
                        {'params': model.W, 'lr': lrN},
                        {'params': model.Bias, 'lr': lrB}])

alossBest = 1e6
epochs = 200
sig   = 0.3
ndata = 3 #n_data_total
bestModel = model
hist = torch.zeros(epochs)

print('         Design       Coords      Reg           gradW       gradKo        gradKc       gradB')
for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    amis = 0.0
    amisb = 0.0
    for i in range(ndata):

        Z, Coords, M = utils.getIterData(S, Aind, Yobs, MSK, i, device=device)
        M = torch.ger(M, M)

        optimizer.zero_grad()
        # From Coords to Seq
        Zout, Zold = model(Coords)
        PSSMpred = F.softshrink(Zout[:20, :].abs(), Zout.abs().mean().item() / 5)
        misfit = utils.kl_div(PSSMpred, Z[:20, :], weight=True)

        # From Seq to Coord
        Cout, CoutOld = model.backwardProp(Z)
        misfitBackward = utils.dRMSD(Cout,Coords,M)

        R = model.NNreg()
        C0 = torch.norm(Cout - CoutOld) ** 2 / torch.numel(Z)
        Z0 = torch.norm(Zout - Zold) ** 2 / torch.numel(Z)
        loss = misfit + misfitBackward + R + C0 + Z0

        loss.backward()

        aloss += loss.detach()
        amis += misfit.detach().item()
        amisb += misfitBackward.detach().item()

        gW  = model.W.grad.norm().item()
        gKo = model.Kopen.grad.norm().item()
        gKc = model.Kclose.grad.norm().item()
        gB  = model.Bias.grad.norm().item()

        optimizer.step()
        # scheduler.step()
        nprnt = 1
        if (i + 1) % nprnt == 0:
            amis = amis / nprnt
            amisb = amisb / nprnt
            print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E" %
                  (j, i, amis, amisb, R.item(),gW,gKo,gKc,gB))
            amis = 0.0
            amisb = 0.0
        # Validation on 0-th data
        if (i + 1) % 500 == 0:
            with torch.no_grad():
                misVal  = 0
                misbVal = 0
                AQdis   = 0
                nVal = len(STesting)
                for jj in range(nVal):
                    # Z, Coords, M = getIterData(SVal, AindVal, YobsVal, MSKVal, jj,device=device)
                    Z, Coords, M = utils.getIterData(STesting, AindTesting, YobsTesting, MSKTesting, jj, device=device)
                    M = torch.ger(M, M)
                    Zout, Zold = model(Coords)
                    PSSMpred = F.softshrink(Zout[:20, :].abs(), Zout.abs().mean().item() / 5)
                    misfit = utils.kl_div(PSSMpred, Z[:20, :], weight=True)

                    misVal += misfit
                    # From Seq to Coord
                    Cout, CoutOld = model.backwardProp(Z)

                    misfitBackward = utils.dRMSD(Cout,Coords,M)
                    # misfitBackward = torch.norm((M*DMt-M*DM))**2/torch.norm((M*DMt))**2

                    misbVal += misfitBackward
                    AQdis += torch.sqrt(misfitBackward)

                print("%2d       %10.3E   %10.3E   %10.3E" % (j, misVal / nVal, misbVal / nVal, AQdis / nVal))
                print('===============================================')

    hist[j] = (aloss).item() / (ndata)
    if aloss < alossBest:
        alossBest = aloss
        bestModel = model


