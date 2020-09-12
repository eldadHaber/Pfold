import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from src import networks

def getRotMat(t):
    A1 = torch.tensor([[np.cos(t[0]), -np.sin(t[0]), 0],
          [np.sin(t[0]), np.cos(t[0]),  0],
           [0, 0, 1]])

    A2 = torch.tensor([[np.cos(t[1]), 0, np.sin(t[1])],[0, 1, 0],
                        [-np.sin(t[1]), 0, np.cos(t[1])]])

    A3 = torch.tensor([[1, 0, 0],[0, np.cos(t[2]), -np.sin(t[2])],[0, np.sin(t[2]), np.cos(t[2])]])

    A = A1@A2@A3

    return A


X = torch.randn(10,5)
Q = getRotMat(torch.randn(3))

Xo = X[:,:3]@Q

Xr, Xco, R = networks.rotatePoints(X, Xo)