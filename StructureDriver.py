import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pnetProcess
import networks
import optimizeNet
import getStructure

dataFile ='./Data/testing'
id, seq, pssm2, entropy, dssp, RN, RCa, RCb, mask = pnetProcess.parse_pnet(dataFile)
L2np = pnetProcess.list2np()

idx     = 5
# get the coordinates of the 5th idx-protein
Sh, S, E, rN, rCa, rCb, msk = L2np(seq[idx], pssm2[idx], entropy[idx],
                                   RN[idx], RCa[idx], RCb[idx], mask[idx])
rN  = torch.tensor(rN)
rCa = torch.tensor(rCa)
rCb = torch.tensor(rCb)
msk = torch.tensor(msk)

mskF = msk*0+1
#msk = msk != 0


Dmm, Dma, Dmb, Dmn, M = pnetProcess.convertCoordToDistMaps(rN, rCa, rCb, mask=msk)
#OMEGA, THETA, PHI = pnetProcess.convertCoordToAngles(rN, rCa, rCb, M)
OMEGA, THETA, PHI = pnetProcess.convertCoordToAnglesVec(rN, rCa, rCb, mask=M)

rN  = rN.double()
rCa = rCa.double()
rCb = rCa.double()
M   = M.double()
Dmm = Dmm.double()

# Get the distance square
rM  = (rN + rCa + rCb)/3.0
rM  = rM - rM[0,:]

Dmm = getStructure.dist(rM).double()

iter = 1500
tol  = 1e-4
Xo   = 1e3*torch.rand(Dmm.shape[0],3)
Xo   = Xo.double()
Xo, Dc = getStructure.getXfromD(Xo, Dmm, M=1.0, niter=iter, tol=tol)

err = torch.norm((Dc-Dmm))/torch.norm(Dmm)
print('Relative distance error  ', err.item())
Xo = Xo - Xo[0,:]
rM = rM - rM[0,:]

R, tmp  = torch.solve(Xo.t()@rM, Xo.t()@Xo)
Xo = Xo@R

errX = torch.norm((Xo-rM))/torch.norm(rM)
print('Relative X error  ', errX.item())
