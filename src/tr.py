import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from src import networks
from src import pnetProcess
from src import utils

pnetProcess.getProteinDataLinear(seq, pssm2, entropy, RN, RCa, RCb, mask, idx, ncourse)


def getRotMat(t):
    A1 = torch.tensor([[np.cos(t[0]), -np.sin(t[0]), 0],
          [np.sin(t[0]), np.cos(t[0]),  0],
           [0, 0, 1]])

    A2 = torch.tensor([[np.cos(t[1]), 0, np.sin(t[1])],[0, 1, 0],
                        [-np.sin(t[1]), 0, np.cos(t[1])]])

    A3 = torch.tensor([[1, 0, 0],[0, np.cos(t[2]), -np.sin(t[2])],[0, np.sin(t[2]), np.cos(t[2])]])

    A = A1@A2@A3

    return A


X = torch.randn(10,3)
Q = getRotMat(torch.randn(3))

Xo = X@Q

Xr, Xco, R = utils.rotatePoints(X, Xo)

a = utils.getRotDist(X, Xo)
print(a)