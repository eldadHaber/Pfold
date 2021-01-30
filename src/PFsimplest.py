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


def getIterData(S, Aind, Yobs, MSK, i, device='cpu'):
    scale = 1e-2
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

    A = torch.zeros(20, n)
    A[a, torch.arange(0, n)] = 1.0
    Seq = torch.cat((PSSM, A))
    Seq = Seq.to(device=device, non_blocking=True)

    Coords = Coords.to(device=device, non_blocking=True)
    M = M.type('torch.FloatTensor')
    M = M.to(device=device, non_blocking=True)

    return Seq, Coords, M


nstart  = 40
nopen   = 128
stsz    = 55
nclose  = 3

Arch = [nstart, nopen, stsz, nclose]

model = networks.simpleNet(Arch)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ',total_params)

lrO = 1e-4
lrC = 1e-4
lrW = 1e-4
lrB = 1e-4


optimizer = optim.Adam([{'params': model.Kopen, 'lr': lrO},
                        {'params': model.Kclose, 'lr': lrC},
                        {'params': model.W, 'lr': lrW},
                        {'params': model.Bias, 'lr': lrB}])

alossBest = 1e6
epochs = 300
sig   = 0.3
ndata = 2000 #n_data_total
bestModel = model
hist = torch.zeros(epochs)
kk = 1

print('         Design       Coords              gradKo        gradKc')
for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    amis = 0.0
    for i in range(ndata):

        Z, Coords, M = getIterData(S, Aind, Yobs, MSK, i, device=device)
        Z = Z.unsqueeze(0)
        Coords = Coords.unsqueeze(0)
        M = M.unsqueeze(0).unsqueeze(0)

        optimizer.zero_grad()
        # From Coords to Seq
        Cout = model(Z)
        d = torch.sqrt(torch.sum((Coords[:, :, 1:] - Coords[:, :, :-1]) ** 2, dim=1)).mean()
        Cout = utils.distConstraint(Cout, d).unsqueeze(0)

        DM = utils.getDistMat(Cout.squeeze(0))
        DMt = utils.getDistMat(Coords.squeeze(0))
        MM = torch.ger(M.squeeze(), M.squeeze())
        misfit = torch.norm(MM * DMt - MM * DM) ** 2 / torch.norm(MM * DMt) ** 2

        loss = misfit

        loss.backward()

        aloss += loss.detach()
        amis += misfit.detach().item()

        gKopen = model.Kopen.grad.norm().item()
        gKclose = model.Kclose.grad.norm().item()

        optimizer.step()
        # scheduler.step()
        nprnt = 20
        if (i + 1) % nprnt == 0:
            amis = amis / nprnt
            print("%2d.%1d   %10.3E   %10.3E   %10.3E " %
                  (j, i, amis, gKopen, gKclose))
            amis = 0.0
            amisb = 0.0

        # Validation on 0-th data
        if i % 100 == 0:
            with torch.no_grad():
                misVal = 0
                AQdis = 0
                # nVal    = len(SVal)
                nVal = len(STesting)
                for jj in range(nVal):
                    Z, Coords, M = getIterData(STesting, AindTesting, YobsTesting, MSKTesting, jj, device=device)
                    Coords = Coords.unsqueeze(0)
                    Z = Z.unsqueeze(0)
                    M = M.unsqueeze(0).unsqueeze(0)

                    Cout = model(Z)
                    d = torch.sqrt(torch.sum((Coords[:, :, 1:] - Coords[:, :, :-1]) ** 2, dim=1)).mean()
                    Cout = utils.distConstraint(Cout, d).unsqueeze(0)

                    MM = torch.ger(M.squeeze(), M.squeeze())
                    DM = utils.getDistMat(Cout.squeeze(0))
                    DMt = utils.getDistMat(Coords.squeeze(0))
                    misfit = torch.norm(MM * DMt - MM * DM) ** 2 / torch.norm(MM * DMt) ** 2

                    misVal += misfit
                    AQi =  torch.norm(MM * (DM - DMt)) / (1.0*MM.shape[0])
                    #print(MM.shape[0], AQi)
                    AQdis += torch.norm(MM * (DM - DMt)) / torch.sqrt(1.0*torch.sum(MM > 0))

            print("%2d       %10.3E   %10.3E" % (j, misVal / nVal, AQdis / nVal))
            print('===============================================')

    hist[j] = (aloss).item() / (ndata)
