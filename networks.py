import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv2(X, Kernel):
    return F.conv2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def conv2T(X, Kernel):
    return F.conv_transpose2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def initVnetParams(A,device='cpu'):
    # A = [ inChan, OutChan, number of layers in this res, ConvSize]
    print('Initializing network  ')
    nL = A.shape[1]
    K = []
    npar = 0
    cnt = 1
    for i in range(nL+1):
        for j in range(A[i,2]):
            if A[i,1] == A[i,0]:
                stdv = 1e-3
            else:
                stdv = 1e-2*A[i,0]/A[i,1]

            Ki = torch.zeros(A[i,1],A[i,0],A[i,3],A[i,3])
            Ki.data.uniform_(-stdv, stdv)
            Ki = nn.Parameter(Ki)
            print('layer number', cnt, 'layer size', Ki.shape[0],Ki.shape[1],Ki.shape[2],Ki.shape[3])
            cnt += 1
            npar += np.prod(Ki.shape)
            Ki.to(device)
            K.append(Ki)

    W = nn.Parameter(torch.randn(4, 64, 1, 1))
    npar += W.numel()
    print('Number of parameters  ', npar)
    return K, W

class vnet2D(nn.Module):
    """ VNet """
    def __init__(self, K, W, h):
        super(vnet2D, self).__init__()
        self.K = K
        self.W = W
        self.h = h

    def forward(self, x):
        """ Forward propagation through the network """

        # Number of layers
        nL = len(self.K)

        # Store the output at different scales to add back later
        xS = []

        # Opening layer
        z = conv2(x, self.K[0])
        z = F.instance_norm(z)
        x = F.relu(z)

        # Step through the layers (down cycle)
        for i in range(1, nL):

            # First case - Residual blocks
            # (same number of input and output kernels)

            sK = self.K[i].shape

            if sK[0] == sK[1]:
                z  = conv2(x, self.K[i])
                z  = F.instance_norm(z)
                z  = F.relu(z)
                z  = conv2T(z, self.K[i])
                x  = x - self.h*z

            # Change number of channels/resolution
            else:
                # Store the features
                xS.append(x)

                z  = conv2(x, self.K[i])
                z  = F.instance_norm(z)
                x  = F.relu(z)

                # Downsample by factor of 2
                x = F.avg_pool2d(x, 3, stride=2, padding=1)

        # Number of scales being computed (how many downsampling)
        n_scales = len(xS)

        # Step back through the layers (up cycle)
        for i in reversed(range(1, nL)):

            # First case - Residual blocks
            # (same number of input and output kernels)
            sK = self.K[i].shape
            if sK[0] == sK[1]:
                z  = conv2T(x, self.K[i])
                z  = F.instance_norm(z)
                z  = F.relu(z)
                z  = conv2(z, self.K[i])
                x  = x - self.h*z

            # Change number of channels/resolution
            else:
                n_scales -= 1
                # Upsample by factor of 2
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

                z  = conv2T(x, self.K[i])
                z  = F.instance_norm(z)
                x  = F.relu(z) + xS[n_scales]

        x = conv2(x, self.W)
        return x

def misfitFun(Ypred, Yobs, Active=torch.tensor([1])):
    n = Yobs.shape
    R = torch.zeros(n)
    R[0, 0, :, :] = (Ypred[0, 0, :, :] + Ypred[0, 0, :, :].t())/2 - Yobs[0, 0, :, :]
    R[0, 1, :, :] = (Ypred[0, 1, :, :] + Ypred[0, 1, :, :].t()) / 2 - Yobs[0, 1, :, :]
    R[0, 2, :, :] =  Ypred[0, 2, :, :]  - Yobs[0, 2, :, :]
    R[0, 3, :, :] =  Ypred[0, 3, :, :] -  Yobs[0, 3, :, :]
    R             = Active*R
    loss  = 0.5*torch.norm(R)**2
    loss0 = 0.5*torch.norm(Active*Yobs)**2
    return loss/loss0

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

