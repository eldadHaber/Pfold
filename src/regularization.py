import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def TVreg(I, Active=torch.tensor([1]), h=(1.0,1.0), eps=1e-3):
    n = I.shape
    IntNormGrad = 0
    normGrad = torch.zeros(n)
    for i in range(n[1]):
        Ix, Iy             =  getGradImage2D(I[:,i,:,:].unsqueeze(1), h)
        normGrad[:,i,:,:]  =  Active * getFaceToCellAv2D(Ix**2, Iy**2)
        IntNormGrad        += torch.sum(torch.sqrt(normGrad[:,:,i,:]+eps))

    return IntNormGrad, normGrad

def getCellToFaceAv2D(I):
    a = torch.tensor([0.5,0.5])
    Ax = torch.zeros(1,1,2,1); Ax[0,0,:,0] = a
    Ay = torch.zeros(1,1,1,2); Ay[0,0,0,:] = a

    Ix = F.conv2d(I,Ax,padding=(1,0))
    Iy = F.conv2d(I,Ay,padding=(0,1))

    return Ix, Iy

def getGradImage2D(I, h=(1.0, 1.0)):
    s = torch.tensor([-1, 1.0])
    Kx = torch.zeros(1, 1, 2, 1);
    Kx[0, 0, :, 0] = s / h[0]
    Ky = torch.zeros(1, 1, 1, 2);
    Ky[0, 0, 0, :] = s / h[1]

    Ix = F.conv2d(I, Kx, padding=(1, 0))
    Iy = F.conv2d(I, Ky, padding=(0, 1))

    return Ix, Iy


def getDivField2D(Ix, Iy, h=(1.0, 1.0)):
    s = torch.tensor([-1, 1.0])
    Kx = torch.zeros(1, 1, 2, 1);
    Kx[0, 0, :, 0] = s / h[0]
    Ky = torch.zeros(1, 1, 1, 2);
    Ky[0, 0, 0, :] = s / h[1]

    Ixx = F.conv_transpose2d(Ix, Kx, padding=(1, 0))
    Iyy = F.conv_transpose2d(Iy, Ky, padding=(0, 1))

    return Ixx + Iyy

def getFaceToCellAv2D(Ix, Iy):
    a = torch.tensor([0.5, 0.5])
    Ax = torch.zeros(1, 1, 2, 1);
    Ax[0, 0, :, 0] = a
    Ay = torch.zeros(1, 1, 1, 2);
    Ay[0, 0, 0, :] = a

    # Average
    Ixa = F.conv_transpose2d(Ix, Ax, padding=(1, 0))
    Iya = F.conv_transpose2d(Iy, Ay, padding=(0, 1))

    return Ixa + Iya
