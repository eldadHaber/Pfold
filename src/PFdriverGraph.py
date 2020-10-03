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
    Xtrue = Yobs[i][0, 0, :n, :n]
    M = MSK[i]
    M = torch.ger(M[:n], M[:n])
    X0 = torch.zeros(3, n)
    X0[0, :] = torch.linspace(0, 1, n)

    # initialization
    Z = torch.zeros(23, n)
    Z[:20, :] = 0.5*Seq
    Z[20:, :] = X0
    return Z, Xtrue, M


nopen = 128
nlayers = 50
Arch = [nopen, nlayers]

model = networks.gNNC(Arch)
total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ',total_params)

lr = 1e-2
sig = 0.2
optimizer = optim.Adam([{'params': model.K, 'lr': lr},{'params': model.W, 'lr': lr}], lr=lr)

alossBest = 1e6
ndata = len(S)-1
epochs = 500
sig = 0.2

ndata = 4
bestModel = model
hist = torch.zeros(epochs)
for j in range(epochs):
    # Prepare the data
    aloss = 0
    for i in range(1,ndata):
    #for i in range(0, 1):

        Z, Xtrue, M = getIterData(S, Yobs, MSK, i)

        optimizer.zero_grad()
        Zout = model(Z, sig)

        D = torch.exp(-utils.getDistMat(Zout)/sig)
        Dt = torch.exp(-utils.getDistMat(Xtrue)/sig)
        loss = torch.norm(M*Dt-M*D)**2/torch.norm(M*Dt)**2
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        aloss += torch.sqrt(loss).detach()
        optimizer.step()
        print(i, '   ', j, '     ', torch.sqrt(loss).item())
    if aloss < alossBest:
        alossBest = aloss
        bestModel = model

    # Validation on 0-th data
    with torch.no_grad():
        Z, Xtrue, M = getIterData(S, Yobs, MSK, 0)
        Zout = model(Z)
        D = torch.exp(-utils.getDistMat(Zout) / sig)
        Dt = torch.exp(-utils.getDistMat(Xtrue) / sig)
        lossV = torch.norm(M*Dt - M*D)**2 / torch.norm(M * Dt)**2

    print('==== Epoch =======',j, '        ', (aloss).item()/(ndata),'   ',lossV.item())
    hist[j] = (aloss).item()/(ndata)

Z, Xtrue, M = getIterData(S, Yobs, MSK, 0)
Zout = model(Z,sig)
D    = torch.exp(-utils.getDistMat(Zout)/sig)
Dt   = torch.exp(-utils.getDistMat(Xtrue)/sig)

print('loss = ', alossBest.item()/(ndata))
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(D.detach())
plt.subplot(1,2,2)
plt.imshow(M*Dt)
plt.figure(2)
plt.plot(hist)
