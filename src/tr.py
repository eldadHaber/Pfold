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
from src import graphUnetworks as gunts


# Test Unet
nLevels  = 4
nin      = 40
nsmooth  = 2
nopen    = 64
nLayers  = 18
nout     = 3
h        = 0.1

#model = gunts.GraphUnet(nL,nIn,nsmooth)
model = gunts.stackedGraphUnet(nLevels,nsmooth,nin,nopen,nLayers,nout,h)
x     = torch.randn(1,40,256)
m     = torch.ones(1, 1,256)

z, zold = model(x, m)

'''
# load Testing data
AindTesting = torch.load('../../../data/casp11/AminoAcidIdxTesting.pt')
YobsTesting = torch.load('../../../data/casp11/RCalphaTesting.pt')
MSKTesting  = torch.load('../../../data/casp11/MasksTesting.pt')
STesting     = torch.load('../../../data/casp11/PSSMTesting.pt')

def getIterData(S, Aind, Yobs, MSK, i, device='cpu',pad=0):
    scale = 1e-3
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
    #PSSM = augmentPSSM(PSSM, 0.01)

    A = torch.zeros(20, n)
    A[a, torch.arange(0, n)] = 1.0
    Seq = torch.cat((PSSM, A))
    Seq = Seq.to(device=device, non_blocking=True)

    Coords = Coords.to(device=device, non_blocking=True)
    M = M.type('torch.FloatTensor')
    M = M.to(device=device, non_blocking=True)

    if pad > 0:
        L = Coords.shape[1]
        k = 2**torch.tensor(L, dtype=torch.float64).log2().round().int()
        k = k.item()
        CoordsPad = torch.zeros(3,k)
        CoordsPad[:,:Coords.shape[1]] = Coords
        SeqPad    = torch.zeros(Seq.shape[0],k)
        SeqPad[:,:Seq.shape[1]] = Seq
        Mpad      = torch.zeros(k)
        Mpad[:M.shape[0]] = M
        M = Mpad
        Seq = SeqPad
        Coords = CoordsPad

    return Seq, Coords, M

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
Z, Coords, M = getIterData(STesting, AindTesting, YobsTesting, MSKTesting, 10, device=device, pad=1)

n0 = 128
sc = 3
A = torch.tensor([128,256,512,1024])
nin = 40
nopen = 64
nout =  3
nL = 10
h = 0.1

model = networks.stackedUnet1D(A,nin,nopen,nout,nL)

#Z = torch.randn(1,40,128)
#m = torch.ones(1,128)
Z = Z.unsqueeze(0)
M = M.unsqueeze(0)
Y = model(Z,M)

Zh = model.backProp(Y,M)
'''