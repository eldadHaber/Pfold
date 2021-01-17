import torch
import torch.nn as nn
import torch.nn.functional as F

from supervised.network_transformer import tr2DistSmall


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

#### Module Graph Convolution Neural Networks ######


class hyperNet(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size=3):
        super(hyperNet, self).__init__()
        Kopen, Kclose, W, Bias = self.init_weights(chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size=stencil_size)
        self.Kopen  = Kopen
        self.Kclose = Kclose
        self.W = W
        self.Bias = Bias
        self.h = 0.1

    def init_weights(self, chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size=3, channels_close=None):
        if channels_close is None:
            channels_close = 2 * channels
        nlayers = nblocks * nlayers_pr_block

        Kopen = torch.zeros(channels, chan_in)
        stdv = 1 * Kopen.shape[0]/Kopen.shape[1]
        Kopen.data.uniform_(-stdv, stdv)
        Kopen = nn.Parameter(Kopen)

        Kclose = torch.zeros(chan_out, channels)
        stdv = 1 * Kclose.shape[0] / Kclose.shape[1]
        Kclose.data.uniform_(-stdv, stdv)
        Kclose = nn.Parameter(Kclose)

        W = torch.zeros(nlayers, 2, channels_close, channels, 9)
        stdv = 1e-1
        W.data.uniform_(-stdv, stdv)
        W = nn.Parameter(W)

        Bias = torch.rand(nlayers,2,channels,1)*1e-3
        Bias = nn.Parameter(Bias)

        return Kopen, Kclose, W, Bias

    def doubleSymLayer(self, Z, Wi, Bi, L, mask):
        Ai0 = conv1((Z + Bi[0]), Wi[0]) * mask
        # Ai0_old = F.instance_norm(Ai0)

        tmp = mean_masked(Ai0, mask, d=2, keepdim=True)
        Ai0 = (Ai0 - tmp) * mask
        Ai0 = Ai0 / torch.sqrt(torch.sum(Ai0 ** 2, dim=1, keepdim=True) + 1e-3)

        Ai1 = (conv1((Z + Bi[1]), Wi[1]) @ L) * mask

        tmp = mean_masked(Ai1, mask, d=2, keepdim=True)
        Ai1 = (Ai1 - tmp) * mask
        Ai1 = Ai1 / torch.sqrt(torch.sum(Ai1 ** 2, dim=1, keepdim=True) + 1e-3)
        # Ai1 = F.instance_norm(Ai1)
        Ai0 = torch.relu(Ai0)
        Ai1 = torch.relu(Ai1)

        # Layer T
        Ai0 = conv1T(Ai0, Wi[0]) * mask
        Ai1 = (conv1T(Ai1, Wi[1]) @ L) * mask
        Ai = Ai0 + Ai1

        return Ai

    def forward(self, Z, mask=None):
        if mask is None:
            mask = torch.ones_like(Z[:,0,:])
        mask = mask.unsqueeze(1)

        h = self.h
        l = self.W.shape[0]
        Kopen = self.Kopen
        Kclose = self.Kclose

        Z = Kopen @ Z * mask
        L, D = getGraphLap(Z,mask=mask, sigma=10)
        for i in range(l):
            if i%10==0:
                L, D = getGraphLap(Z,mask=mask, sigma=10)

            Wi = self.W[i]
            Bi = self.Bias[i]
            # Layer
            Ai = self.doubleSymLayer(Z, Wi, Bi, L, mask)
            Z = Z - h * Ai
        # closing layer back to desired shape
        Z = Kclose @ Z * mask

        dists = ()
        for i in range(Z.shape[1]//3):
            dists += (tr2DistSmall(Z[:, i * 3:(i + 1) * 3, :]),)

        return dists, Z

    def NNreg(self):
        dWdt = self.W[1:] - self.W[:-1]
        RW   = torch.sum(torch.abs(dWdt))/dWdt.numel()
        RKo  = torch.norm(self.Kopen)**2/2/self.Kopen.numel()
        RKc = torch.norm(self.Kclose)**2/2/self.Kclose.numel()
        return RW + RKo + RKc














class gNNC(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size=3):
        super(gNNC, self).__init__()
        K0, K, W, Wend = self.init_weights(chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size=stencil_size)
        self.K = K
        self.K0 = K0
        self.Wend = Wend


        self.W = W
        self.h = 0.01

    def init_weights(self, chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size=3):
        K0 = torch.zeros(channels, chan_in)
        stdv = 1e-1 * channels / chan_in
        K0.data.uniform_(-stdv, stdv)
        K0 = nn.Parameter(K0)


        nlayers = nlayers_pr_block*nblocks
        Knet = torch.zeros(nlayers, chan_out)
        stdv = 1e-1 * channels / chan_in
        Knet.data.uniform_(-stdv, stdv)
        Knet = nn.Parameter(Knet)

        W = torch.zeros(nlayers, 3, channels, channels, 1)
        stdv = 1e-1
        W.data.uniform_(-stdv, stdv)
        W = nn.Parameter(W)

        Wend = nn.Parameter(1e-4 * torch.randn(chan_out, channels, 1))
        return K0, Knet, W, Wend

    def forward(self, Z, mask=None):
        # mask = None
        if mask is None:
            mask = torch.ones_like(Z[:,0,:])
        mask = mask.unsqueeze(1)

        h = self.h
        l = self.W.shape[0]
        K0 = self.K0
        Knet = self.K
        # opening layer
        # Z = K0 @ Z
        Z = torch.tanh(K0@Z) * mask
        for i in range(l):
            # Compute the graph
            L, D = getGraphLap(Z, mask)
            # L = torch.ones_like(L)
            Ki = Knet[i]
            Wi = self.W[i]

            # Layer
            Ai0 = Ki[0] * (conv1(Z, Wi[0]) * mask)
            Ai1 = Ki[1] * (conv1(Z, Wi[1]) * mask) @ L
            Ai2 = (Ki[2] * (conv1(Z, Wi[2]) * mask) @ L) @ L
            Ai = Ai0 + Ai1 + Ai2

            tmp = mean_masked(Ai, mask, d=2, keepdim=True)
            Ai = (Ai - tmp) * mask
            Ai = Ai / torch.sqrt(torch.sum(Ai ** 2, dim=1, keepdim=True) + 1e-3)
            Ai = torch.relu(Ai)

            # Layer T
            Ai0 = Ki[0] * (conv1T(Ai, Wi[0]) * mask)
            Ai1 = Ki[1] * (conv1T(Ai, Wi[1]) * mask) @ L
            Ai2 = (Ki[2] * (conv1T(Ai, Wi[2]) * mask) @ L) @ L
            Ai = Ai0 + Ai1 + Ai2

            Z = Z - h * Ai

        Z = conv1(Z, self.Wend) * mask

        dists = ()
        for i in range(Z.shape[1]//3):
            dists += (tr2DistSmall(x[:, i * 3:(i + 1) * 3, :]),)

        return dists, Z



def getGraphLap(X, mask, sigma=None):
    # normalize the data
    mm = (mask.float().transpose(1,2) @ mask.float()).int()

    X = X - mean_masked(X, mask, d=2,keepdim=True) * mask
    X = X / torch.sqrt(torch.sum(X**2,dim=2,keepdim=True) / torch.sum(mask, dim=2, keepdim=True) + 1e-4)


    # add  position vector
    pos = 0.5 * torch.linspace(0, 1, X.shape[2], device=X.device)[None,:] * mask
    Xa = torch.cat((X, pos), dim=1)
    W = tr2DistSmall(Xa)
    W = W * mm
    if sigma is None:
        sigma = mean_masked(W, mm, d=(1,2), keepdim=True)
    W = torch.exp(-W/sigma*10) * mm
    # D = torch.diag(torch.sum(W, dim=1))
    Wsum = torch.sum(W, dim=1)

    E = torch.eye(Wsum.size(1),device=X.device)
    tmp = Wsum.unsqueeze(2).expand(*Wsum.size(), Wsum.size(1))
    D = tmp * E

    L = D - W
    L = 0.5 * (L + L.transpose(1,2))
    #
    # Dd = torch.zeros_like(Wsum)
    # Dd = Dd.flatten()
    # m1 = (mask == 1).flatten()
    # nominator = (torch.sqrt(torch.diagonal(D, dim1=1, dim2=2)) + 1e-4).flatten()
    # Dd[m1] = 1 / nominator[m1]
    # Dd=Dd.reshape(Wsum.shape)
    # # Dd2 = (1 / (torch.sqrt(torch.diagonal(D, dim1=1, dim2=2)) + 1e-4)) * mask.squeeze()
    # tmp = Dd.unsqueeze(2).expand(*Dd.size(), Wsum.size(1))
    # Dh = tmp * E
    # # Dh = torch.diag(1/torch.sqrt(torch.diag(D)));
    # Lnorm = Dh @ L @ Dh
    # Lnorm = 0.5 * (Lnorm + Lnorm.transpose(1,2))
    # #
    # b = 1
    # import matplotlib.pyplot as plt
    # plt.spy(L[b,:,:].cpu().detach())
    # plt.title("L")
    # plt.figure()
    # plt.imshow(L[b,:,:].cpu().detach())
    # plt.title("L")
    # plt.figure()
    # plt.spy(D[b,:,:].cpu().detach())
    # plt.title("D")
    # plt.figure()
    # plt.imshow(D[b,:,:].cpu().detach())
    # plt.title("D")
    # plt.figure()
    # plt.spy(W[b,:,:].cpu().detach())
    # plt.title("W")
    # plt.figure()
    # plt.imshow(W[b,:,:].cpu().detach())
    # plt.title("W")
    # plt.figure()
    # plt.spy(Dh[b,:,:].cpu().detach())
    # plt.title("Dh")
    # plt.figure()
    # plt.imshow(Dh[b,:,:].cpu().detach())
    # plt.title("Dh")
    # plt.figure()
    # plt.spy(Lnorm[b,:,:].cpu().detach())
    # plt.title("Lnorm")
    # plt.figure()
    # plt.imshow(Lnorm[b,:,:].cpu().detach())
    # plt.title("Lnorm")
    #
    # plt.pause(1)


    return L, W

def mean_masked(X,mask,d,keepdim=False):
    """
    Will find the mean of a tensor, X, along dimension, d, with the mask, mask, applied to the tensor.
    """
    Xm = torch.sum(X*mask, dim=d, keepdim=keepdim) / torch.sum(mask, dim=d, keepdim=keepdim)
    # Xm = Xm * mask  # (N, L, C)
    return Xm
