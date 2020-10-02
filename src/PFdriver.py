from src import proTerp
from src import pnetProcess
from src import utils
from src import networks
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

dataFile ='./../data/testing'
id, seq, pssm2, entropy, dssp, RN, RCa, RCb, mask = pnetProcess.parse_pnet(dataFile)

idx     = np.arange(0,3)
ncourse = 4
X = []; Yobs = []; M = []; Sh = []
for i in idx:
    Xi, Yobsi, Mi, Shi, Si, rMi = pnetProcess.getProteinData(seq, pssm2, entropy, RN, RCa, RCb, mask, i, ncourse)
    X.append(Xi)
    Yobs.append(Yobsi/5000)
    M.append(Mi)
    Sh.append(Shi)
    
    print('Image ',i, 'imsize ', Xi.shape)


# Initialize a network
nc = 32
A = torch.tensor([[400,  nc,  1, 3],
                  [nc,  nc,  5, 3],
                  [nc, 2*nc,  1, 3]])

dvc = 'cpu'
vnet    = networks.vnet2D(A, 0.01)

ii = 0
Ypred     = vnet(X[ii].unsqueeze(0))

optimizer = optim.Adam(vnet.parameters(), lr=1e-3)
epochs     = 300

numdat = 3
ehist = torch.zeros(epochs)
hist = torch.zeros(numdat)
for epoch in range(epochs):
    for i in range(numdat-1):
        # Get a protein
        input = X[i].unsqueeze(0)

        vnet.zero_grad()
        outputFake = vnet(input)
        Dfake = torch.exp(-outputFake[0,0,:,:])
        nx = Dfake.shape[0]
        Dfake = Dfake - torch.diag(torch.diag(Dfake)) + torch.eye(nx,nx)
        Dtrue = torch.exp(-Yobs[i][0,0,:,:])
        loss = F.mse_loss(M[i][0]*Dfake, M[i][0]*Dtrue)/F.mse_loss(M[i][0]*Dtrue, 0*Dtrue)
        #misfit = alpha*errGD + (1-alpha)*errGC
        #if errGD >5:
        #    error
        loss.backward()
        optimizer.step()
        hist[i] = torch.sqrt(loss.detach()).item()
        print('[%d / %d][%d / %d]  Loss: %.4f '
              % (epoch, epochs, i,numdat,  torch.sqrt(loss)))

    print('   ')
    print('======== EPOCH %d  avmisfit = %.4f ================'%(epoch, torch.mean(hist)))
    print('   ')
    ehist[epoch] = torch.mean(hist)
    hist = 0 * hist


input = X[numdat].unsqueeze(0)

vnet.zero_grad()
outputFake = vnet(input)
Dfake = torch.exp(-outputFake[0, 0, :, :])
nx = Dfake.shape[0]
Dfake = Dfake - torch.diag(torch.diag(Dfake)) + torch.eye(nx, nx)
Dtrue = torch.exp(-Yobs[numdat][0, 0, :, :])
loss = F.mse_loss(M[numdat][0] * Dfake, M[numdat][0] * Dtrue) / F.mse_loss(M[numdat][0] * Dtrue, 0 * Dtrue)

plt.figure(1)
plt.plot(ehist)
plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(M[numdat][0]*Dtrue)
plt.subplot(1,2,2)
plt.imshow(M[numdat][0]*Dfake.detach())
print(loss)