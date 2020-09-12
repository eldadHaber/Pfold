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
def conv1T(X, Kernel):
    return F.conv_transpose1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

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

    W = nn.Parameter(1e-4*torch.randn(4, 64, 1, 1))
    npar += W.numel()
    print('Number of parameters  ', npar)
    return K, W

class vnet1D(nn.Module):
    """ VNet """
    def __init__(self, Arch,nout):
        super(vnet1D, self).__init__()
        K, W = self.init_weights(Arch,nout)
        self.K = K
        self.W = W
        self.h = 0.1

    def init_weights(self,A,nout):
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

    def forward(self, x):
        """ Forward propagation through the network """

        # Number of layers
        nL = len(self.K)

        # Store the output at different scales to add back later
        xS = []

        # Opening layer
        z = conv1(x, self.K[0])
        z = F.instance_norm(z)
        x = F.relu(z)

        # Step through the layers (down cycle)
        for i in range(1, nL):

            # First case - Residual blocks
            # (same number of input and output kernels)

            sK = self.K[i].shape

            if sK[0] == sK[1]:
                z  = conv1(x, self.K[i])
                z  = F.instance_norm(z)
                z  = F.relu(z)
                z  = conv1T(z, self.K[i])
                x  = x - self.h*z

            # Change number of channels/resolution
            else:
                # Store the features
                xS.append(x)

                z  = conv1(x, self.K[i])
                z  = F.instance_norm(z)
                x  = F.relu(z)

                # Downsample by factor of 2
                x = F.avg_pool1d(x, 3, stride=2, padding=1)

        # Number of scales being computed (how many downsampling)
        n_scales = len(xS)

        # Step back through the layers (up cycle)
        for i in reversed(range(1, nL)):

            # First case - Residual blocks
            # (same number of input and output kernels)
            sK = self.K[i].shape
            if sK[0] == sK[1]:
                z  = conv1T(x, self.K[i])
                z  = F.instance_norm(z)
                z  = F.relu(z)
                z  = conv1(z, self.K[i])
                x  = x - self.h*z

            # Change number of channels/resolution
            else:
                n_scales -= 1
                # Upsample by factor of 2
                x = F.interpolate(x, scale_factor=2)

                z  = conv1T(x, self.K[i])
                z  = F.instance_norm(z)
                x  = F.relu(z) + xS[n_scales]

        x = conv1(x, self.W)
        return x


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

        output = self.transformer_encoder(src, self.src_mask)
        output = output - torch.mean(output, dim=0).unsqueeze(0)

        output = self.reshapeF(output)
        output = self.decoder(output)
        output = self.reshapeB(output)
        return output

######

def tr2Dist(Y):

    k = Y.shape[2]
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

def rotatePoints(X, Xo):
    # Find a matrix R such that Xo@R = X
    # Translate X to fit Xo
    # (X+c)R = V*S*V' R + c*R = Xo
    # X = Uo*So*Vo'*R' - C

    # R = V*inv(S)*U'
    if X.shape != Xo.shape:
        U, S, V =  torch.svd(X)
        S[3:] = 0
        X = U@torch.diag(S)@V.t()
        X = X[:,:3]

    n, dim = X.shape

    Xc  = X - X.mean(dim=0)
    Xco = Xo - Xo.mean(dim=0)

    C = (Xc.t()@Xco) / n

    U, S, V = torch.svd(C)
    d = torch.sign((torch.det(U) * torch.det(V)))

    R  = V@torch.diag(torch.tensor([1.0,1,d],dtype=U.dtype))@U.t()

    Xr = Xc@R.t()
    print(torch.norm(Xco - Xc @ R.t()))

    return Xr, Xco, R

def getRotDist(Xc, Xo, alpha = 1.0):

    Xr, Xco, R = rotatePoints(Xc, Xo)
    Do = torch.sum(Xo**2,dim=1,keepdim=True) + torch.sum(Xo**2,dim=1,keepdim=True).t() - 2*Xo@Xo.t()
    Do = torch.sqrt(torch.relu(Do))
    Dc = torch.sum(Xc**2,dim=1,keepdim=True) + torch.sum(Xc**2,dim=1,keepdim=True).t() - 2*Xc@Xc.t()
    Dc = torch.sqrt(torch.relu(Dc))

    return F.mse_loss(Xr, Xco) + alpha*F.mse_loss(Dc, Do)