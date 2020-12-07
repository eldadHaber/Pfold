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

#X    = torch.load('../../../data/casp12Testing/Xtest.pt')
Yobs = torch.load('../../../data/casp12Testing/Coordtest.pt')
MSK  = torch.load('../../../data/casp12Testing/Masktest.pt')
S     = torch.load('../../../data/casp12Testing/Seqtest.pt')

def getIterData(S,Yobs,MSK,i):
    Seq = S[i].t()
    n = Seq.shape[1]
    M = MSK[i][:n]

    X = Yobs[i][0, 0, :n, :n]
    X = utils.linearInterp1D(X,M)
    X = torch.tensor(X)
    #D = torch.sum(torch.pow(X,2), dim=0, keepdim=True) + torch.sum(torch.pow(X,2), dim=0, keepdim=True).t() - 2*X.t()@X
    #D = 0.5*(D+D.t())
    #mm = torch.diag(M)
    #D  = mm @ D @ mm
    U, Lam, V = torch.svd(X)

    #C = torch.zeros(10,n)
    #C[:5,:] = torch.diag(torch.sqrt(Lam[:5]))@U[:,:5].t()
    #C[5:,:] = torch.diag(torch.sqrt(Lam[:5]))@V[:, :5].t()
    Coords = torch.diag(Lam)@V.t()
    Coords = Coords.type('torch.FloatTensor')
    return Seq, Coords, M


nstart = 3
nopen = 32*2
nhid  = 64*2
nclose = 20
nlayers = 72
h       = 1/nlayers
Arch = [nstart, nopen, nhid, nclose, nlayers]

model = networks.graphNN(Arch)
total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ',total_params)

lrO = 1e-4/2
lrC = 1e-4/4
lrN = 1e-2/4
lrB = 1e-2/2

optimizer = optim.Adam([{'params': model.Kopen, 'lr': lrO},
                        {'params': model.Kclose, 'lr': lrC},
                        {'params': model.W, 'lr': lrN},
                        {'params': model.Bias, 'lr': lrB}])

alossBest = 1e6
ndata = len(S)-1
epochs = 500

ndata = 40
bestModel = model
hist = torch.zeros(epochs)
for j in range(epochs):
    # Prepare the data
    aloss = 0
    for i in range(1,ndata):

        Z, Coords, M = getIterData(S, Yobs, MSK, i)
        optimizer.zero_grad()
        Zout = model(Coords)
        #print('network output norm ',Zout.norm().detach().item())
        onehotZ = torch.argmax(Z, dim=0)
        wloss = torch.zeros(20)
        for kk in range(20):
            wloss[kk] = torch.sum(onehotZ==kk)
        wloss = 1.0/(wloss+1.0)
        misfit = F.cross_entropy(Zout.t(), onehotZ,wloss)
        R    = model.graphNNreg()
        loss = misfit + 0.5*R
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        aloss += loss.detach()
        optimizer.step()
        print(j,'.',i,'   ', misfit.item(),'   ',R.item(),'   ',
                model.W.grad.norm().item(),'   ',
                model.Kopen.grad.norm().item(),'   ',
                model.Kclose.grad.norm().item(),'   ',
                model.Bias.grad.norm().item())
    if aloss < alossBest:
        alossBest = aloss
        bestModel = model

    # Validation on 0-th data
    with torch.no_grad():
        Z, Coords, M = getIterData(S, Yobs, MSK, 0)
        onehotZ = torch.argmax(Z, dim=0)
        Zout = model(Coords)
        lossV = F.cross_entropy(Zout.t(), onehotZ)
    print('==== Epoch =======',j, '        ', (aloss).item()/(ndata-1),'   ',lossV.item())
    hist[j] = (aloss).item()/(ndata)

