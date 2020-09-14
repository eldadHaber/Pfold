import os, sys
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pylab as P
import pnetProcess
from scipy import interpolate


# Example
dataFile = './../data/testing'
L2np = pnetProcess.list2np()
id, seq, pssm2, entropy, dssp, RN, RCa, RCb, mask = pnetProcess.parse_pnet(dataFile)
rCa = RCa[0]
rCa = T.tensor(L2np(rCa))
D0 = T.sqrt(T.relu(T.sum(rCa**2, dim=2) + T.sum(rCa**2, dim=2).t() - 2*rCa.squeeze(0)@rCa.squeeze(0).t()))
P.figure(1)
P.imshow(D0)

M   = T.tensor(L2np(mask[0]))
x = T.linspace(0, M.shape[1] - 1, M.shape[1])
M = M[0,:]
x = x[M != 0]
xs = T.linspace(0, M.shape[0] - 1, M.shape[0])
y = rCa[0,:,:]
y = y[M != 0,:]

f = interpolate.interp1d(x, y, kind='linear', axis=0, copy=True, bounds_error=False, fill_value="extrapolate")
ys = f(xs)
rCa[0,:,:] = T.tensor(ys)

D1 = T.sqrt(T.relu(T.sum(rCa ** 2, dim=2) + T.sum(rCa ** 2, dim=2).t() - 2 * rCa.squeeze(0) @ rCa.squeeze(0).t()))

P.figure(2)
P.imshow(D1)

    #P.scatter(x, y, label='Samples', color='purple')
    #P.plot(xs, ys, label='Interpolated curve')
    #P.show()