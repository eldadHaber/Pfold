import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
from src import networks, optimizeNet, pnetProcess
import matplotlib.pyplot as plt
import torch.optim as optim
from src import utils
from src import regularization as rg
import torch.nn.functional as F

##################################################
# The potential is \sum_{ij} p(|r_i-r_j|) = p(D, theta)
# where D is the distance matrix
# based on 1d convolutions
# dp/dD = 0

def potentialEnergy(D,K):

    h = 0.1
    for i in len(K):
        D = D - networks.conv1DT(torch.relu(networks.conv1D(D, K[i])),K[i])

    return torch.sum(D)

A = torch.tensor([1,16,32,64,128,256])
K = []
for i in range(len(A)-1):
    Ki = torch.randn(A[i+1],A[i],1,1)
    K.append(Ki)

n = 50
def minPotential(D,K,lr):

    optimizer = optim.Adam([{'params': D, 'lr': lr}], lr=lr)
    for i in range(50):
        optimizer.zero_grad()
        p = potentialEnergy(D, K)
        p.backward()
        optimizer.step()

    return D
