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
Yobs = torch.load('../../../data/casp12Testing/Coordtest.pt')
MSK  = torch.load('../../../data/casp12Testing/Masktest.pt')
S     = torch.load('../../../data/casp12Testing/Seqtest.pt')

def getIterData(S,Yobs,MSK,i,device='cpu'):
    Seq = S[i].t()
    n = Seq.shape[1]
    M = MSK[i][:n]

    X = Yobs[i][0, 0, :n, :n]
    X = utils.linearInterp1D(X,M)
    X = torch.tensor(X)

    X = X - torch.mean(X,dim=1, keepdim=True)
    U, Lam, V = torch.svd(X)

    Coords = torch.diag(Lam)@V.t()
    Coords = Coords.type('torch.FloatTensor')
    Seq    = Seq.type(torch.float32)
    Seq    = Seq.to(device=device, non_blocking=True)
    Coords = Coords.to(device=device, non_blocking=True)
    M      = M.to(device=device, non_blocking=True)

    return Seq, Coords, M


nstart  = 3
nopen   = 32
nhid    = 64
nclose  = 20
nlayers = 50
h       = 1/nlayers
Arch = [nstart, nopen, nhid, nclose, nlayers]

#model = networks.graphNN(Arch)
model = networks.hyperNet(Arch)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ',total_params)

lrO = 1e-3
lrC = 1e-3
lrN = 1e-4
lrB = 1e-4

optimizer = optim.Adam([{'params': model.Kopen, 'lr': lrO},
                        {'params': model.Kclose, 'lr': lrC},
                        {'params': model.W, 'lr': lrN},
                        {'params': model.Bias, 'lr': lrB}])

alossBest = 1e6
epochs = 3
sig   = 0.2
ndata = 3
bestModel = model
hist = torch.zeros(epochs)
print('         Design       Coords      Reg           gradW       gradKo        gradKc       gradB')
for j in range(epochs):
    # Prepare the data
    aloss = 0
    for i in range(1,ndata):

        Z, Coords, M = getIterData(S, Yobs, MSK, i, device=device)
        M = torch.ger(M,M)

        optimizer.zero_grad()
        # From Coords to Seq
        Zout, Zold = model(Coords)
        onehotZ = torch.argmax(Z, dim=0)
        wloss = torch.zeros(20).to(device)
        for kk in range(20):
            wloss[kk] = torch.sum(onehotZ==kk)
        wloss = 1.0/(wloss+1.0)

        misfit = F.cross_entropy(Zout.t(), onehotZ,wloss)
        # From Seq to Coord
        Cout, CoutOld  = model.backwardProp(Z)
        D = torch.exp(-utils.getDistMat(Cout)/sig)
        Dt = torch.exp(-utils.getDistMat(Coords)/sig)
        misfitBackward = torch.norm(M*Dt-M*D)**2/torch.norm(M*Dt)**2

        R    = model.NNreg()
        C0   = torch.norm(Cout - CoutOld)**2/torch.numel(Z)
        Z0   = torch.norm(Zout-Zold)**2/torch.numel(Z)
        loss = misfit + misfitBackward + R + C0 + Z0

        loss.backward(retain_graph=True)
        #torch.nn.utils.clip_grad_norm_(model.W, 0.1)
        #torch.nn.utils.clip_grad_norm_(model.Kclose, 0.1)
        #torch.nn.utils.clip_grad_norm_(model.Kopen, 0.1)
        #torch.nn.utils.clip_grad_norm_(model.Bias, 0.1)

        aloss += loss.detach()
        optimizer.step()

        print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E" %
              (j,i,misfit.item(), misfitBackward.item(), R.item(),
               model.W.grad.norm().item(),model.Kopen.grad.norm().item(),model.Kclose.grad.norm().item(),
               model.Bias.grad.norm().item()))
    if aloss < alossBest:
        alossBest = aloss
        bestModel = model

    # Validation on 0-th data
    #with torch.no_grad():
    #    Z, Coords, M = getIterData(S, Yobs, MSK, 0)
    #    onehotZ = torch.argmax(Z, dim=0)
    #    Zout = model(Coords)
    #    lossV = F.cross_entropy(Zout.t(), onehotZ)
    #print('==== Epoch =======',j, '        ', (aloss).item()/(ndata-1),'   ',lossV.item())
    #hist[j] = (aloss).item()/(ndata)

