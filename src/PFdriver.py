import numpy as np
import torch
from src import networks, optimizeNet, pnetProcess
import matplotlib.pyplot as plt

dataFile ='./../data/testing'
id, seq, pssm2, entropy, dssp, RN, RCa, RCb, mask = pnetProcess.parse_pnet(dataFile)

idx     = np.arange(0,1)
ncourse = 4
X = []; Yobs = []; M = []; Sh = []
for i in idx:
    Xi, Yobsi, Mi, Shi, Si = pnetProcess.getProteinData(seq, pssm2, entropy, RN, RCa, RCb, mask, i, ncourse)
    X.append(Xi)
    Yobs.append(Yobsi)
    M.append(Mi)
    Sh.append(Shi)
    
    print('Image ',i, 'imsize ', Xi.shape)

# Initialize a network
A = torch.tensor([[21,   64,   1, 3],
                  [64,   64,   5, 3],
                  [64,   128,  1, 3],
                  [128,  128,  5, 3],
                  [128,  256,  1, 3],
                  [256,  256,  5, 3],
                  [256,  512,  1, 3],
                  [512,  512,  5, 3],
                  [512, 1024,  1, 3]])

dvc = 'cpu'
K, W  = networks.initVnetParams(A, device=dvc)
VN    = networks.vnet2D(K, W, 0.01)

ii = 0
Ypred     = VN(X[ii])

loss = networks.misfitFun(Ypred, Yobs[ii], M[ii])
print('Initial misfit ', loss.item())

reg, normGrad = networks.TVreg(Ypred, M[ii])
print('Initial reg ', reg.item())
iters    = 1000 #40*1500
lr       = [1e-3, 1e-1]
rp       = 1e-4
dweights = torch.tensor([0.1, 1.0, 1.0, 1.0])

#A = Yobs[0][201:312,201:312]
#S, As = QuadTree.createQuadTreeFromImage(A, 19)

VN, hist = optimizeNet.trainNetwork(VN, X, Yobs, M, iters, lr, regpar = rp, dweights = dweights)

Ypred     = VN(X[0])

pnetProcess.plotProteinData(Yobs[0], 1)
pnetProcess.plotProteinData(M[0] * Ypred.detach(), 2)

#R = nn.Parameter(torch.randn(X.shape[2],3))
#D = Yobs[0,0,:,:]
#R = optimizeNet.getCoordsFromDist(R,D, lr=1e-2,niter=100)