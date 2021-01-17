import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from supervised import utils


def conv2(X, Kernel):
    return F.conv2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def conv1(X, Kernel):
    return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))
def conv1T(X, Kernel):
    return F.conv_transpose1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def conv2T(X, Kernel):
    return F.conv_transpose2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def tv_norm(X, eps=1e-3):
    X = X - torch.mean(X,dim=1, keepdim=True)
    X = X/torch.sqrt(torch.mean(X,dim=1,keepdim=True) + eps)
    return X


def getDistMat(X, msk=torch.tensor([1.0])):
    D = torch.sum(torch.pow(X, 2), dim=0, keepdim=True) + torch.sum(torch.pow(X, 2), dim=0,
                                                                    keepdim=True).t() - 2 * X.t() @ X

    dev = X.device
    msk = msk.to(dev)

    mm = torch.ger(msk, msk)
    return mm * torch.sqrt(torch.relu(D))


def getGraphLap(X,sig=10):

    # normalize the data
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / (torch.std(X, dim=1, keepdim=True) + 1e-3)
    #X = X/torch.sqrt(torch.sum(X**2,dim=1,keepdim=True)/X.shape[1] + 1e-4)
    # add  position vector
    pos = torch.linspace(-0.5, 0.5, X.shape[1]).unsqueeze(0)
    dev = X.device
    pos = pos.to(dev)
    Xa = torch.cat((X, 5e1*pos), dim=0)
    W = getDistMat(Xa)
    W = torch.exp(-W/sig)
    D = torch.diag(torch.sum(W, dim=0))
    L = D - W
    Dh = torch.diag(1/torch.sqrt(torch.diag(D)));
    L = Dh @ L @ Dh

    L = 0.5 * (L + L.t())

    return L, W

class hyperNet(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, Arch):
        super(hyperNet, self).__init__()
        Kopen, Kclose, W, Bias = self.init_weights(Arch)
        self.Kopen  = Kopen
        self.Kclose = Kclose
        self.W = W
        self.Bias = Bias
        self.h = 0.1

    def init_weights(self,A):
        print('Initializing network  ')
        #Arch = [nstart, nopen, nhid, nclose, nlayers]
        nstart = A[0]
        nopen  = A[1]
        nhid   = A[2]
        nclose = A[3]
        nlayers = A[4]

        Kopen = torch.zeros(nopen, nstart)
        stdv = 1e-3 * Kopen.shape[0]/Kopen.shape[1]
        Kopen.data.uniform_(-stdv, stdv)
        Kopen = nn.Parameter(Kopen)

        Kclose = torch.zeros(nclose, nopen)
        stdv = 1e-3 * Kclose.shape[0] / Kclose.shape[1]
        Kclose.data.uniform_(-stdv, stdv)
        Kclose = nn.Parameter(Kclose)

        W = torch.zeros(nlayers, 2, nhid, nopen, 9)
        stdv = 1e-4
        W.data.uniform_(-stdv, stdv)
        W = nn.Parameter(W)

        Bias = torch.rand(nlayers,2,nopen,1)*1e-4
        Bias = nn.Parameter(Bias)

        return Kopen, Kclose, W, Bias

    def doubleSymLayer(self, Z, Wi, Bi, L):
        Ai0 = conv1((Z + Bi[0]).unsqueeze(0), Wi[0])
        Ai0 = F.instance_norm(Ai0)
        Ai1 = (conv1((Z + Bi[1]).unsqueeze(0), Wi[1]).squeeze(0)@L).unsqueeze(0)
        Ai1 = F.instance_norm(Ai1)
        Ai0 = torch.relu(Ai0)
        Ai1 = torch.relu(Ai1)

        # Layer T
        Ai0 = conv1T(Ai0, Wi[0])
        Ai1 = (conv1T(Ai1, Wi[1]).squeeze(0) @ L.t()).unsqueeze(0)
        Ai = Ai0 + Ai1

        return Ai

    def forward(self, Z, m=1.0):

        h = self.h
        l = self.W.shape[0]
        Kopen = self.Kopen
        Kclose = self.Kclose

        Z = Kopen@Z
        Zold = Z
        L, D = utils.getGraphLap(Z)
        for i in range(l):
            if i%10==0:
                L, D = utils.getGraphLap(Z)

            Wi = self.W[i]
            Bi = self.Bias[i]
            # Layer
            Ai = self.doubleSymLayer(Z, Wi, Bi, L)
            Ztemp = Z
            Z = 2*Z - Zold - (h**2)*Ai.squeeze(0)
            Zold = Ztemp
            # for non hyperbolic use
            # Z = Z - h*Ai.squeeze(0)
        # closing layer back to desired shape
        Z    = Kclose@Z
        Zold = Kclose@Zold
        return Z, Zold

    def backwardProp(self, Z):

        h = self.h
        l = self.W.shape[0]

        Kopen = self.Kopen
        Kclose = self.Kclose

        # opening layer
        Z = Kclose.t()@Z
        Zold = Z
        L, D = utils.getGraphLap(Z)
        for i in reversed(range(l)):
            if i%10==0:
                L, D = utils.getGraphLap(Z)
            Wi = self.W[i]
            Bi = self.Bias[i]
            Ai = self.doubleSymLayer(Z, Wi, Bi, L)

            Ztemp = Z
            Z = 2*Z - Zold - (h**2)*Ai.squeeze(0)
            Zold = Ztemp

        # closing layer back to desired shape
        Z    = Kopen.t()@Z
        Zold = Kopen.t()@Zold
        return Z, Zold


    def NNreg(self):

        dWdt = self.W[1:] - self.W[:-1]
        RW   = torch.sum(torch.abs(dWdt))/dWdt.numel()
        RKo  = torch.norm(self.Kopen)**2/2/self.Kopen.numel()
        RKc = torch.norm(self.Kclose)**2/2/self.Kclose.numel()
        return RW + RKo + RKc

