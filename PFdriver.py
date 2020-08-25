import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pnetProcess
import networks
import optimizeNet

dataFile ='./Data/testing'
id, seq, pssm2, entropy, dssp, RN, RCa, RCb, mask = pnetProcess.parse_pnet(dataFile)

idx     = 5
ncourse = 3
X, Yobs, M = pnetProcess.getProteinData(seq, pssm2, entropy, RN, RCa, RCb, mask, idx, ncourse)

# Initialize a network
A = torch.tensor([[21,   64,   1, 3],
                  [64,   64,   5, 3],
                  [64,   128,  1, 3],
                  [128,  128,  5, 3],
                  [128,  256,  1, 3],
                  [256,  256,  5, 3],
                  [256,  512,  1, 3]])

dvc = 'cpu'
K, W  = networks.initVnetParams(A,device=dvc)
VN    = networks.vnet2D(K, W, 0.01)

Ypred     = VN(X)

loss = networks.misfitFun(Ypred, Yobs, M)
print('Initial misfit ', loss.item())

reg, normGrad = networks.TVreg(Ypred, M)
print('Initial reg ', reg.item())
iters    = 1500
lr       = [1e-3, 1e-1]
rp       = 1e-4
dweights = torch.tensor([0.1, 1.0, 1.0, 1.0])

VN, hist = optimizeNet.trainNetwork(VN, X, Yobs, M, iters, lr, regpar = rp, dweights = dweights)

Ypred     = VN(X)

pnetProcess.plotProteinData(Yobs,1)
pnetProcess.plotProteinData(M*Ypred.detach(),2)

#R = nn.Parameter(torch.randn(X.shape[2],3))
#D = Yobs[0,0,:,:]
#R = optimizeNet.getCoordsFromDist(R,D, lr=1e-2,niter=100)