import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def conv2(X, Kernel):
    return F.conv2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def conv1(X, Kernel):
    return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def conv2T(X, Kernel):
    return F.conv_transpose2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def initVnetParams(A,device='cpu'):
    # A = [ inChan, OutChan, number of layers in this res, ConvSize]
    print('Initializing network  ')
    nL = A.shape[0]
    K = []
    npar = 0
    cnt = 1
    for i in range(nL):
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

def misfitFun(Ypred, Yobs, Active=torch.tensor([1]), dweights = torch.tensor([1,1,1,1.0])):
    n = Yobs.shape
    dweights = dweights.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    R = torch.zeros(n)
    W = 1.0/(Yobs + Yobs.mean()/100)
    R[0, 0, :, :] = ((Ypred[0, 0, :, :] + Ypred[0, 0, :, :].t())/2 - Yobs[0, 0, :, :])
    R[0, 1, :, :] = ((Ypred[0, 1, :, :] + Ypred[0, 1, :, :].t())/2 - Yobs[0, 1, :, :])
    R[0, 2, :, :] = ((Ypred[0, 2, :, :] + Ypred[0, 2, :, :].t())/2 - Yobs[0, 2, :, :])
    R[0, 3, :, :] = ((Ypred[0, 3, :, :] + Ypred[0, 3, :, :].t())/2 -  Yobs[0, 3, :, :])
    R             = Active*(dweights*W*R)
    loss  = 0.5*torch.norm(R)**2
    loss0 = 0.5*torch.norm(Active*(dweights*W*Yobs))**2
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

##### Transformers =====

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, ntokenOut=-1):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Conv1d(ntoken, ninp, 7, padding=3) #nn.Linear(ntoken, ninp)
        self.ninp = ninp
        if ntokenOut < 0:
            ntokenOut = ntoken
        self.decoder = nn.Conv1d(ninp, ntokenOut, 7, padding=3) #nn.Linear(ninp, ntokenOut)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        #src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.encoder(src[0,:,:].t().unsqueeze(0)).squeeze(0).t().unsqueeze(0) * math.sqrt(self.ninp)
        #src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        #output = self.decoder(output)
        output = self.decoder(output[0, :, :].t().unsqueeze(0)).squeeze(0).t().unsqueeze(0)
        return output


def tr2Dist(Y):

    k = Y.shape[2]
    #Z = Y[0,:,:]
    #D = torch.sum(Z ** 2, dim=1).unsqueeze(0) + torch.sum(Z ** 2, dim=1).unsqueeze(1) - 2 * Z @ Z.t()
    #D = D/Z.shape[1]
    D = 0.0
    for i in range(k):
        Z = Y[:,:,i]
        Z = Z - torch.mean(Z,dim=1,keepdim=True)
        D = D + torch.sum(Z**2,dim=1).unsqueeze(0) + torch.sum(Z**2,dim=1).unsqueeze(1) - 2*Z@Z.t()
    D = D/k
    return D

def tr2DistSmall(Y):

    k = Y.shape[2]
    Z = Y[0,:,:]
    Z = Z - torch.mean(Z, dim=0, keepdim=True)
    D = torch.sum(Z**2, dim=1).unsqueeze(0) + torch.sum(Z**2, dim=1).unsqueeze(1) - 2*Z@Z.t()
    D = 3*D/k
    return torch.sqrt(torch.relu(D))

##### Hamiltonian Networks

def inithNetParams(A,device='cpu'):
    # A = [ inChan, OutChan, number of layers]
    print('Initializing network  ')
    stdv = 1e-2
    Wopen = torch.zeros(A[0,0], A[0,1])
    Wopen.data.uniform_(-stdv, stdv)
    Wopen  = nn.Parameter(Wopen)
    Wclose = torch.zeros(A[2,0], A[2,1])
    Wclose.data.uniform_(-stdv, stdv)
    Wclose = nn.Parameter(Wclose)

    K    = torch.zeros(A[1,0], A[1,1], A[1,2])
    stdv = 1e-3 * A[1, 0]/A[1, 1]
    K.data.uniform_(-stdv, stdv)
    K = nn.Parameter(K)
    npar = K.numel() + Wopen.numel() + Wclose.numel()
    K.to(device)

    print('Number of parameters  ', npar)
    return K, Wopen, Wclose


def chnorm(F, eps=1e-3):
    F = F - torch.mean(F,dim=1,keepdim=True)
    F = F/(torch.sqrt(torch.sum(F**2,dim=1) + eps).unsqueeze(1))
    return F

class hNet(nn.Module):
        # Solve the ODE
        # x_{j+1} = x_j + relu(K_j*z_j) + relu(W*Forceing_j)
        # z_{j+1} = z_j - relu(K_j'*x_{j+1})
        def __init__(self, K, Wopen, Wclose, h):
            super(hNet, self).__init__()
            self.K      = K
            self.Wopen  = Wopen
            self.Wclose = Wclose
            self.h     = h

        def forward(self, Frc):
            Nseq     = Frc.shape[2]# length of sequence
            NfeatIn  = self.K.shape[1]
            NfeatHid = self.K.shape[0]
            NHidLyr  = self.K.shape[2]

            Nbatch   = Frc.shape[0]

            # allocate space for output sequence
            Y   = torch.zeros(Nbatch,self.Wclose.shape[0], Nseq)
            x   = torch.zeros(Nbatch,NfeatIn)
            z   = torch.zeros(Nbatch,NfeatHid)
            for i in range(Nseq):
                Fi  = Frc[:, :, i]
                if i+1<Nseq:
                    Fip = Frc[:, :, i+1]
                else:
                    Fip = 0.0
                for j in range(NHidLyr):
                    Fi = ((NHidLyr-j)*Fi + j*Fip)/NHidLyr
                    #if j==0:
                    fi = torch.relu(chnorm(F.linear(Fi, self.Wopen)))
                    #else:
                    #    fi = 0.0

                    aj = torch.relu(chnorm(F.linear(x, self.K[:,:,j])))
                    z  = z + self.h*(aj + fi)
                    qi = torch.relu(chnorm(F.linear(z, self.K[:,:,j].t())))
                    x  = x - self.h*qi

                Y[:,:,i] = F.linear(z, self.Wclose)

            return Y

def Seq2Dist(Y):

    D = torch.sum(Y**2,dim=1) + torch.sum(Y**2,dim=1).t() -  Y[0,:,:].t()@Y[0,:,:]
    return D