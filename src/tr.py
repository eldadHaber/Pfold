import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from src import networks
from src import pnetProcess
from src import utils
import matplotlib.pylab as P
from src import networks

def h_poly(t):
    n = torch.linspace(0, 3, 4)
    tt = t**n
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]])
    return A@tt


def interp(x, y, xs):
  m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
  m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
  I = P.searchsorted(x[1:], xs)
  dx = (x[I+1]-x[I])
  hh = h_poly((xs-x[I])/dx)
  return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx


# Example
#if __name__ == "__main__":
#  x = torch.linspace(0, 6, 7)
#  y = x.sin()
#  xs = torch.linspace(0, 6, 101)
#  ys = interp(x, y, xs)
#  P.scatter(x, y, label='Samples', color='purple')
#  P.plot(xs, ys, label='Interpolated curve')
#  P.plot(xs, xs.sin(), '--', label='True Curve')
#  P.legend()
#  P.show()

jj = 1
while flag:
    Xi = X[jj]
    Yi = Yobs[jj][:, 0, :, :]
    Mi = MSK[jj]
    input = torch.zeros(1, 24, Xi.shape[2])
    input[:, :21, :] = Xi
    input[:, 21:, :] = Yi
    # try to obtain in-out patches
    patchIn, patches = utils.getRandomCrop(input, Mi, winsize=winsize, batchSize=[32])
    if len(patchIn) > 0:
        # patchOut =  patchIn[:, 21:, :]
        patchOut = patches[:, 21:, :]
        flag = False
    else:
        jj += 1
# Now train the generator
maskSize = torch.randint(16, 32, (1,)).item()
randomMask = utils.getRandomMask(maskSize, winsize)
mm = torch.ones(24, 128);
mm[21:, :] = randomMask
# patchOutFake = netG(randomMask*patchIn, randomMask)
patchesIn = mm * patches
patchOutFake = netG(patchesIn, randomMask)

X = patchOut[16,:,:]; Xf = patchOutFake[16,:,:].detach()

Df = torch.sqrt(torch.relu(torch.sum(Xf**2,dim=0,keepdim=True) + torch.sum(Xf**2,dim=0,keepdim=True).t() - 2*Xf.t()@Xf))
D = torch.sqrt(torch.relu(torch.sum(X**2,dim=0,keepdim=True) + torch.sum(X**2,dim=0,keepdim=True).t() - 2*X.t()@X))
plt.figure(1)
plt.imshow(Df)
plt.colorbar()
plt.figure(2)
plt.imshow(D)
plt.colorbar()
mmm = randomMask.unsqueeze(1)@randomMask.unsqueeze(0)
plt.figure(3)
plt.imshow(mmm*D)
plt.colorbar()
plt.figure(4)
plt.imshow(torch.abs(D-Df))
plt.colorbar()
