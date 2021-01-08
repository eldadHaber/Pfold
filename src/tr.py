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

def pairedDist(X,Y):
        return torch.sum((X-Y)**2,dim=0)

def distConstraint(X1,X2, X3):
        d11 = pairedDist(X1[:, 0:], X1[:, :-1])
        d22 = pairedDist(X2[:, 0:], X2[:, :-1])
        d33 = pairedDist(X3[:, 0:], X3[:, :-1])
        d12 = pairedDist(X1, X2)
        d13 = pairedDist(X1, X3)
        d23 = pairedDist(X2, X3)








