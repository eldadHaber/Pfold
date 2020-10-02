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
Seq   = S[0].t()
n = Seq.shape[1]
Xtrue = Yobs[0][0,0,:n,:n]
M     = MSK[0]

M = torch.ger(M[:n],M[:n])
X0 = torch.zeros(3,n)
X0[0,:] = torch.linspace(-0.3309,0.9591,n)

# initialization
Z = torch.zeros(23,n)
Z[:20,:] = 0.05*Seq
Z[20:,:] = X0


def gNN(Z,K,W):

    n = Seq.shape[1]
    h = 0.1
    l = W.shape[0]
    # opening layer
    Z = K@Z
    for i in range(l):
        # Compute eignevectors of the graph
        L, D = utils.getGraphLap(Z, sig=2)
        #L = L + 0.1*torch.eye(L.shape[0])
        L    = L.t()@L
        Wi = W[i, :, :]

        # Layer
        Z  = Z@L
        Ai = Wi@Z
        Ai = Ai - Ai.mean(dim=0,keepdim=True)
        Ai = Ai/torch.sqrt(torch.sum(Ai**2,dim=0,keepdim=True)+1e-3)
        Z  = Z - h*Wi.t()@torch.relu(Ai)@L.t()

    return Z

nopen = 128
nlayers = 50
K = nn.Parameter(1e-1*torch.rand(nopen,23))
W = nn.Parameter(1e-5*torch.randn(nlayers,128,nopen))
#Z = 0.5*Z
lr = 1e-2
sig = 0.2
optimizer = optim.Adam([{'params': K},{'params': W}], lr=lr)

print('Number of parameters   ',W.numel() + K.numel())

Kbest = K
Wbest = W
for j in range(1000):

    optimizer.zero_grad()
    Zout = gNN(Z,K,W)
    D = torch.exp(-utils.getDistMat(Zout)/sig)
    Dt = torch.exp(-utils.getDistMat(Xtrue)/sig)
    loss = torch.norm(M*Dt-M*D)**2/torch.norm(M*Dt-0*D)**2

    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(K, 0.5)
    torch.nn.utils.clip_grad_norm_(W, 0.5)

    optimizer.step()
    print(j,'     ',torch.sqrt(loss).item())

plt.subplot(1,2,1)
plt.imshow(D.detach())
plt.subplot(1,2,2)
plt.imshow(M*Dt)