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

##### 1D VNET ###########################
#
#
#
class vnet1D(nn.Module):
    """ VNet """
    def __init__(self, Arch,nout,h=0.1):
        super(vnet1D, self).__init__()
        K, W = self.init_weights(Arch,nout)
        self.K = K
        self.W = W
        self.h = h

    def init_weights(self,A,nout):
        print('Initializing network  ')
        nL = A.shape[0]
        K = nn.ParameterList([])
        npar = 0
        cnt = 1
        for i in range(nL):
            for j in range(A[i, 2]):
                if A[i, 1] == A[i, 0]:
                    stdv = 1e-4
                else:
                    stdv = 1e-4 * A[i, 0] / A[i, 1]

                Ki = torch.zeros(A[i, 1], A[i, 0], A[i, 3])
                Ki.data.uniform_(-stdv, stdv)
                Ki = nn.Parameter(Ki)
                print('layer number', cnt, 'layer size', Ki.shape[0], Ki.shape[1], Ki.shape[2])
                cnt += 1
                npar += np.prod(Ki.shape)
                #Ki.to(device)
                K.append(Ki)

        W = nn.Parameter(1e-4*torch.randn(nout, A[0,1], 1))
        npar += W.numel()
        print('Number of parameters  ', npar)
        return K, W

    def forward(self, x, m=1.0):
        """ Forward propagation through the network """

        # Number of layers
        nL = len(self.K)

        # Store the output at different scales to add back later
        xS = []
        mS = [m]
        # Opening layer
        z = mS[-1]*conv1(x, self.K[0])

        z = F.instance_norm(z)
        #z = tv_norm(z)
        x = F.relu(z)

        # Step through the layers (down cycle)
        for i in range(1, nL):

            # First case - Residual blocks
            # (same number of input and output kernels)

            sK = self.K[i].shape

            if sK[0] == sK[1]:
                z  = mS[-1]*conv1(x, self.K[i])
                z  = F.instance_norm(z)
                #z = tv_norm(z)
                z  = F.relu(z)
                z  = mS[-1]*conv1T(z, self.K[i])
                #print('======== A')
                #print(x.norm())
                x  = x - self.h*z
                #print(x.norm())
                #print('======== A')

            # Change number of channels/resolution
            else:
                # Store the features
                xS.append(x)
                z  = mS[-1]*conv1(x, self.K[i])
                z  = F.instance_norm(z)
                #z  = tv_norm(z)
                x  = F.relu(z)

                # Downsample by factor of 2
                x = F.avg_pool1d(x, 3, stride=2, padding=1)
                m = F.avg_pool1d(m.unsqueeze(0), 3, stride=2, padding=1).squeeze(0)
                mS.append(m)
        # Number of scales being computed (how many downsampling)
        n_scales = len(xS)

        # Step back through the layers (up cycle)
        for i in reversed(range(1, nL)):

            # First case - Residual blocks
            # (same number of input and output kernels)
            sK = self.K[i].shape
            if sK[0] == sK[1]:
                z  = mS[-1]*conv1T(x, self.K[i])
                z  = F.instance_norm(z)
                #z = tv_norm(z)
                z  = F.relu(z)
                z  = mS[-1]*conv1(z, self.K[i])
                #print('======== B')
                #print(x.norm())
                x  = x - self.h*z
                #print(x.norm())
                #print('======== B')


            # Change number of channels/resolution
            else:
                n_scales -= 1
                # Upsample by factor of 2
                x = F.interpolate(x, scale_factor=2)
                mS = mS[:-1]
                z  = mS[-1].unsqueeze(0)*conv1T(x, self.K[i])
                z  = F.instance_norm(z)
                #z = tv_norm(z)

                x  = F.relu(z) + xS[n_scales]

        x = conv1(x, self.W)
        return x

#
#
#
##### END 1D VNET ###########################


##### VNET 2D ###########################

class vnet2D(nn.Module):
    """ VNet """
    def __init__(self, A,h):
        super(vnet2D, self).__init__()
        K, W = self.initVnetParams2D(A)
        self.K = K
        self.W = W
        self.h = h

    def initVnetParams2D(self,A, device='cpu'):
        # A = [ inChan, OutChan, number of layers in this res, ConvSize]
        print('Initializing network  ')
        nL = A.shape[0]
        K = nn.ParameterList([])
        npar = 0
        cnt = 1
        for i in range(nL):
            for j in range(A[i, 2]):
                if A[i, 1] == A[i, 0]:
                    stdv = 1e-3
                else:
                    stdv = 1e-2 * A[i, 0] / A[i, 1]

                Ki = torch.zeros(A[i, 1], A[i, 0], A[i, 3], A[i, 3])
                Ki.data.uniform_(-stdv, stdv)
                Ki = nn.Parameter(Ki)
                print('layer number', cnt, 'layer size', Ki.shape[0], Ki.shape[1], Ki.shape[2], Ki.shape[3])
                cnt += 1
                npar += np.prod(Ki.shape)
                Ki.to(device)
                K.append(Ki)

        W = nn.Parameter(1e-4 * torch.randn(1,A[0,1] , 1, 1))
        W = nn.ParameterList([W])
        #npar += W.numel()
        print('Number of parameters  ', npar)
        return K, W

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

        x = conv2(x, self.W[0])
        return x



##### Transformers =====
#
#
#

class CNN(nn.Module):

    def __init__(self, nIn, nhid1, nhid2, nOut, stencilsize):
        super(CNN, self).__init__()

        self.K1 = nn.Conv1d(nIn,   nhid1, stencilsize, padding=stencilsize // 2)
        self.K2 = nn.Conv1d(nhid1, nhid2, stencilsize, padding=stencilsize // 2)
        self.K3 = nn.Conv1d(nhid2, nhid1, stencilsize, padding=stencilsize // 2)
        self.K4 = nn.Conv1d(nhid1, nhid2, stencilsize, padding=stencilsize // 2)
        self.K5 = nn.Conv1d(nhid2, nhid1, stencilsize, padding=stencilsize // 2)
        self.K6 = nn.Conv1d(nhid1, nOut,  stencilsize, padding=stencilsize // 2)
        self.init_weights()

    def init_weights(self):
        initrange  = 0.1
        initrangeR = 0.001

        nn.init.uniform_(self.K1.weight, -initrange, initrange)
        nn.init.uniform_(self.K2.weight, -initrangeR, initrangeR)
        nn.init.uniform_(self.K3.weight, -initrangeR, initrangeR)
        nn.init.uniform_(self.K4.weight, -initrangeR, initrangeR)
        nn.init.uniform_(self.K5.weight, -initrangeR, initrangeR)
        nn.init.uniform_(self.K6.weight, -initrange, initrange)

    def forward(self, src):
        z1 = torch.relu(self.K1(src))
        z2 = z1 + self.K3(torch.relu(self.K2(z1)))
        z3 = z2 + self.K5(torch.relu(self.K4(z1)))
        z3 = self.K6(z3)
        return z3

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout, ntokenOut, stencilsize):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.model_type = 'Transformer'
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        ## Encoder
        #self.encoder = nn.Linear(ntoken, ninp)
        #self.encoder = nn.Conv1d(ntoken, ninp, stencilsize, padding=stencilsize // 2)
        self.encoder = CNN(ntoken, 2*ntoken, 3*ntoken, ninp, stencilsize)

        self.ninp = ninp
        #self.decoder = nn.Linear(ninp, ntoken)
        #self.decoder = nn.Conv1d(ninp, ntokenOut, stencilsize, padding=stencilsize // 2)
        self.decoder = CNN(ninp, 2*ninp, 3*ninp, ntokenOut, stencilsize)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def reshapeF(self,src):
        return src.squeeze(1).t().unsqueeze(0)
    def reshapeB(self,src):
        return src.squeeze(0).t().unsqueeze(1)

    def init_weights(self):
        initrange = 0.1
        #nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        #nn.init.zeros_(self.decoder.weight)
        #nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.reshapeF(src)
        src = self.encoder(src)
        src = self.reshapeB(src)

        #output = self.transformer_encoder(src, self.src_mask)
        outputf = self.transformer_encoder(src, self.src_mask)
        outputb = self.transformer_encoder(torch.flipud(src), self.src_mask)
        output  = outputf + torch.flipud(outputb)
        output = output - torch.mean(output, dim=0).unsqueeze(0)

        output = self.reshapeF(output)
        output = self.decoder(output)
        output = self.reshapeB(output)
        return output




class hyperNet(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, Arch, h=0.1):
        super(hyperNet, self).__init__()
        Kopen, Kclose, W, Bias= self.init_weights(Arch)
        self.Kopen  = Kopen
        self.Kclose = Kclose
        self.W = W
        self.Bias = Bias
        self.h = h

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

    def doubleSymGradLayer(self, Z, Wi, Bi, D):

        Ai0 = conv1((Z + Bi[0]).unsqueeze(0), Wi[0])
        Ai0 = F.instance_norm(Ai0)

        C1  = conv1((Z + Bi[1]).unsqueeze(0), Wi[1]).squeeze(0)
        Ai1 = utils.graphGrad(C1,D).unsqueeze(0)
        Ai1 = F.instance_norm(Ai1)
        Ai0 = torch.relu(Ai0)
        Ai1 = torch.relu(Ai1)

        # Layer T
        Ai0 = conv1T(Ai0, Wi[0])
        C1  = conv1T(Ai1, Wi[1]).squeeze(0)
        Ai1 = utils.graphDiv(C1,D).unsqueeze(0)
        Ai = Ai0 + Ai1

        return Ai

    def forward(self, Z, m=1.0):

        h = self.h
        l = self.W.shape[0]
        Kopen = self.Kopen
        Kclose = self.Kclose
        L, D = utils.getGraphLap(Z)
        Z = Kopen@Z
        Zold = Z
        for i in range(l):
            if i%10==0:
                L, D = utils.getGraphLap(Kclose@Z)

            Wi = self.W[i]
            Bi = self.Bias[i]
            # Layer
            Ai = self.doubleSymLayer(Z, Wi, Bi, L)
            #Ai = self.doubleSymGradLayer(Z, Wi, Bi, D)
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

        L, D = utils.getGraphLap(Z)
        # opening layer
        Z = Kclose.t()@Z
        Zold = Z

        for i in reversed(range(l)):
            if i%10==0:
                L, D = utils.getGraphLap(Kclose@Z)
            Wi = self.W[i]
            Bi = self.Bias[i]
            Ai = self.doubleSymLayer(Z, Wi, Bi, L)
            #Ai = self.doubleSymGradLayer(Z, Wi, Bi, D)
            Ztemp = Z
            Z = 2*Z - Zold - (h**2)*Ai.squeeze(0)
            Zold = Ztemp

        # closing layer back to desired shape
        Z    = (Kopen.t()@Z)
        Zold = (Kopen.t()@Zold)
        return Z, Zold


    def NNreg(self):

        dWdt = self.W[1:] - self.W[:-1]
        RW   = torch.sum(torch.abs(dWdt))/dWdt.numel()
        RKo  = torch.norm(self.Kopen)**2/2/self.Kopen.numel()
        RKc = torch.norm(self.Kclose)**2/2/self.Kclose.numel()
        return RW + RKo + RKc

##### END hyper Convolution Neural Networks ######


