import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from src import utils


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
    X = X/torch.sqrt(torch.sum(X**2,dim=1,keepdim=True) + eps)
    return X

def restrict(A):
    Ac = (A[0:-1:2,0:-1:2] + A[1::2,0:-1:2] + A[0:-1:2,1::2] + A[1::2,1::2])/4.0
    return Ac


def getDistMat(X):
    D = torch.sum(torch.pow(X, 2), dim=0, keepdim=True) + torch.sum(torch.pow(X, 2), dim=0,
                  keepdim=True).t() - 2 * X.t() @ X

    return torch.relu(D) #torch.sqrt(torch.relu(D))

def getGraphLap(X,M=torch.ones(1),sig=10):

    X = X.squeeze(0)
    M = M.squeeze()
    # normalize the data
    #X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / (torch.std(X, dim=1, keepdim=True) + 1e-2)

    W = getDistMat(X)
    We = torch.exp(-W/sig)
    D = torch.diag(torch.sum(We, dim=0))
    L = D - We

    if torch.numel(M)>1:
        MM = torch.ger(M,M)
        L  = MM*L
        II = torch.diag(1-M)
        L  = L+II

    #Dh = torch.diag(1/torch.sqrt(torch.diag(D)))
    #L = Dh @ L @ Dh

    L = 0.5 * (L + L.t())
    return L


class GraphUnet(nn.Module):
    """ VNet """

    def __init__(self, nLevels, nIn, nsmooth):
        super(GraphUnet, self).__init__()
        K = self.init_weights(nLevels, nIn, nsmooth)
        self.K = K

    def init_weights(self, nL, nIn, nsmooth):
        print('Initializing network  ')

        stencil_size = 9
        K = nn.ParameterList([])
        npar = 0
        cnt = 1
        k = nIn
        stdv = 1e-3
        # Kernels the reduce the size
        for i in range(nL):
            # Smoothing layer
            for jj in range(nsmooth):
                Ki = torch.zeros(k, k, stencil_size)
                Ki.data.uniform_(-stdv, stdv)
                Ki = nn.Parameter(Ki)
                print('layer number', cnt, 'layer size', Ki.shape[0], Ki.shape[1], Ki.shape[2])
                cnt += 1
                npar += np.prod(Ki.shape)
                K.append(Ki)

            Ki = torch.zeros(2 * k, k, stencil_size)
            Ki.data.uniform_(-stdv, stdv)
            Ki = nn.Parameter(Ki)
            print('layer number', cnt, 'layer size', Ki.shape[0], Ki.shape[1], Ki.shape[2])
            cnt += 1
            npar += np.prod(Ki.shape)
            K.append(Ki)
            k = 2 * k
        # Smoothing on the coarsest grid kernels
        for i in range(nsmooth):
            Ki = torch.zeros(k, k, stencil_size)
            Ki.data.uniform_(-stdv, stdv)
            Ki = nn.Parameter(Ki)
            print('layer number', cnt, 'layer size', Ki.shape[0], Ki.shape[1], Ki.shape[2])
            cnt += 1
            npar += np.prod(Ki.shape)
            K.append(Ki)

        print('Number of parameters  ', npar)
        return K

    def forward(self, x, X, m=torch.ones(1)):
        """ Forward propagation through the network """

        # Number of layers
        nL = len(self.K)

        # Store the output at different scales to add back later
        xS = []
        mS = [m]
        Xs = [X]
        L = getGraphLap(X, m)
        Ls = [L]
        # Step through the layers (down cycle)
        for i in range(nL):
            coarsen = self.K[i].shape[0] != self.K[i].shape[1]
            if coarsen:
                xS.append(x)

            # print(mS[-1].shape,x.shape)
            z = mS[-1] * conv1(x @ Ls[-1], self.K[i])
            z = F.instance_norm(z)
            if coarsen == False:
                x = x + F.relu(z)
            else:
                x = F.relu(z)

            if coarsen:
                x = F.avg_pool1d(x, 3, stride=2, padding=1)
                m = F.avg_pool1d(m, 3, stride=2, padding=1)
                X = F.avg_pool1d(X, 3, stride=2, padding=1)
                L = getGraphLap(X, m)
                mS.append(m)
                Ls.append(L)

        # Number of scales being computed (how many downsampling)
        n_scales = len(xS)

        # Step back through the layers (up cycle)
        for i in reversed(range(nL)):
            refine = self.K[i].shape[0] != self.K[i].shape[1]
            if refine:
                n_scales -= 1
                # Upsample by factor of 2
                x = F.interpolate(x, scale_factor=2)
                mS = mS[:-1]
                Ls = Ls[:-1]

            # print(mS[-1].shape, x.shape)
            z = mS[-1] * conv1T(x @ Ls[-1], self.K[i])
            z = F.instance_norm(z)
            if refine:
                x = F.relu(z)
            else:
                x = x + F.relu(z)
            if refine:
                x = x + xS[n_scales]

        return x


##### END UNET ###########################

class stackedGraphUnet(nn.Module):

    def __init__(self, nLevels,nsmooth,nin,nopen,nLayers,nout,h=0.1):
        super(stackedGraphUnet, self).__init__()
        Unets, Kopen, Kclose = self.init_weights(nLevels,nsmooth,nin,nopen,nLayers,nout)
        self.Unets = nn.ModuleList(Unets)
        self.h = h
        self.Kopen = Kopen
        self.Kclose = Kclose

    def init_weights(self,nLevels,nsmooth,nin,nopen,nLayers,nout):

        print('Initializing network  ')
        Kopen  = nn.Parameter(torch.rand(nopen, nin)*1e-1)
        Kclose = nn.Parameter(torch.rand(nout, nopen)*1e-1)

        Unets = []
        total_params = 0
        for i in range(nLayers):
            Unet = GraphUnet(nLevels, nopen, nsmooth)
            Unets.append(Unet)
            total_params += sum(p.numel() for p in Unet.parameters())

        print('Total Number of parameters ', total_params)
        return Unets, Kopen, Kclose

    def forward(self, x, m=torch.tensor([1.0])):

        nL = len(self.Unets)

        x  = self.Kopen@x
        xold = x

        for i in range(nL):

            #if i%5==0:
            Coords = self.Kclose@x
            Coords = utils.distConstraint(Coords).unsqueeze(0)
            #x = self.Kclose.t()@X
            temp = x
            Ai = self.Unets[i](x,Coords,m)
            x = 2*x - xold - self.h**2*Ai
            xold = temp

        x = self.Kclose@x
        xold = self.Kclose@xold
        return x, xold

    def backProp(self, x, m=torch.tensor([1.0])):

        nL = len(self.Unets)
        x  = self.Kclose.t()@x
        xold = x
        for i in reversed(range(nL)):
            temp = x
            Coords = self.Kclose@x
            Coords = utils.distConstraint(Coords).unsqueeze(0)
            x = 2*x - xold - self.h**2*self.Unets[i](x,Coords,m)
            xold = temp

        x = self.Kopen.t()@x
        xold = self.Kopen.t()@xold
        return x, xold


##### END 1D Stacked Unet ####################


#X = torch.rand(1,3,8)*10
#X[0,:,5:]=0
#M = torch.ones(1,1,8)
#M[0,0,5:] = 0
#L = getGraphLap(X,M)

#print(L)