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
#from src import networks
from src import graphUnetworks as gunts


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load training data
#Aind = torch.load('../../../data/casp11/AminoAcidIdx.pt')
#Yobs = torch.load('../../../data/casp11/RCalpha.pt')
#MSK  = torch.load('../../../data/casp11/Masks.pt')
#S     = torch.load('../../../data/casp11/PSSM.pt')
# load validation data
#AindVal = torch.load('../../../data/casp11/AminoAcidIdxVal.pt')
#YobsVal = torch.load('../../../data/casp11/RCalphaVal.pt')
#MSKVal  = torch.load('../../../data/casp11/MasksVal.pt')
#SVal     = torch.load('../../../data/casp11/PSSMVal.pt')

# load Testing data
AindTesting = torch.load('../../../data/casp11/AminoAcidIdxTesting.pt')
YobsTesting = torch.load('../../../data/casp11/RCalphaTesting.pt')
MSKTesting  = torch.load('../../../data/casp11/MasksTesting.pt')
STesting     = torch.load('../../../data/casp11/PSSMTesting.pt')


print('Number of elements ', len(AindTesting))

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

a = torch.zeros(len(AindTesting))
for i in range(len(AindTesting)):
#for i in range(30,31):

    Seq, X, M = getIterData(STesting, AindTesting, YobsTesting, MSKTesting, i)


    M = torch.ger(M,M)

    Dtrue = torch.sum(X**2,dim=0,keepdim=True) + torch.sum(X**2,dim=0,keepdim=True).t() - 2*X.t()@X
    Dtrue = M*torch.sqrt(torch.relu(Dtrue))

    #plt.figure(1)
    #plt.imshow(Dtrue)
    #plt.colorbar()

    n = X.shape[1]
    Xl = torch.zeros(3,n)
    Xl[0,:] = 3.8*torch.arange(0,n)

    Dl = torch.sum(Xl**2,dim=0,keepdim=True) + torch.sum(Xl**2,dim=0,keepdim=True).t() - 2*Xl.t()@Xl
    Dl = M*torch.sqrt(torch.relu(Dl))

    #plt.figure(2)
    #plt.imshow(Dl)
    #plt.colorbar()

    R = Dtrue-Dl
    #plt.figure(3)
    #plt.imshow(torch.relu(R))
    #plt.colorbar()
    a[i] = torch.relu(R).sum()
    print(i,a[i].item())
    #print('number of bad elements', torch.sum(R>0))
