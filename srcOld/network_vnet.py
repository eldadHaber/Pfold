import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from srcOld.network_transformer import tr2DistSmall


class ResNet(nn.Module):
    def __init__(self, nlayers,nf_in=526):
        super(ResNet, self).__init__()
        nf = 64
        self.conv_init = nn.Conv2d(nf_in, nf, kernel_size=1)
        blocks = []
        dilations = [1,2,4,8,16]
        for i in range(nlayers):
            dilation = dilations[np.mod(i,len(dilations))]
            block = ResidualBlock(nf,dilation)
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)
        self.activation = nn.ELU()
        # self.dNN = nn.Conv2d(nf, 1, kernel_size=3, padding=1)
        # self.dCaCa = nn.Conv2d(nf, 1, kernel_size=3, padding=1)
        self.dCbCb = nn.Conv2d(nf, 1, kernel_size=3, padding=1)
        # self.dNCa = nn.Conv2d(nf, 1, kernel_size=3, padding=1)
        # self.dNCb = nn.Conv2d(nf, 1, kernel_size=3, padding=1)
        # self.dCaCb = nn.Conv2d(nf, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_init(x)
        for block in self.blocks:
            x = block(x)
        x = self.activation(x)
        # dNCa = self.dNCa(x)
        # dNCb = self.dNCb(x)
        # dCaCb = self.dCaCb(x)
        y = 0.5 * (x + torch.transpose(x, -1, -2))
        # dNN = self.dNN(y)
        # dCaCa = self.dCaCa(y)
        dCbCb = self.dCbCb(y)
        return dCbCb
        # return dNN, dCaCa, dCbCb, dNCa, dNCb, dCaCb


class ResidualBlock(nn.Module):
    def __init__(self, nf,dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(nf,nf, kernel_size=3,dilation=dilation, padding=dilation)
        self.norm1 = nn.InstanceNorm2d(nf)
        self.dropout = nn.Dropout2d()
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, dilation=dilation, padding=dilation)
        self.norm2 = nn.InstanceNorm2d(nf)
        self.activation = nn.ELU()

    def forward(self, x):
        y = self.activation(x)
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.norm2(y)
        z = y + x
        return z

def conv1(X, Kernel):
    return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))
def conv1T(X, Kernel):
    return F.conv_transpose1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))



class vnet1D(nn.Module):
    """ VNet """
    def __init__(self, arch, chan_out):
        super(vnet1D, self).__init__()
        K, W = self.init_weights(arch, chan_out)
        self.K = K
        self.W = W
        self.h = 0.1

    def init_weights(self,A,nout):
        # print('Initializing network  ')
        A = np.array(A)
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
                # print('layer number', cnt, 'layer size', Ki.shape[0], Ki.shape[1], Ki.shape[2])
                cnt += 1
                npar += np.prod(Ki.shape)
                #Ki.to(device)
                K.append(Ki)

        W = nn.Parameter(1e-4*torch.randn(nout, A[0,1], 1))
        npar += W.numel()
        # print('Number of parameters  ', npar)
        return K, W

    def forward(self, x, mask):
        """ Forward propagation through the network """

        mask = mask.unsqueeze(1)

        # Number of layers
        nL = len(self.K)

        # Store the output at different scales to add back later
        xS = []

        # Store the masks at different scales to mask with later
        mS = []

        # Opening layer
        z = conv1(x, self.K[0]) * mask
        z = masked_instance_norm(z,mask)
        x = F.relu(z)

        # Step through the layers (down cycle)
        for i in range(1, nL):

            # First case - Residual blocks
            # (same number of input and output kernels)

            sK = self.K[i].shape

            if sK[0] == sK[1]:
                z = conv1(x, self.K[i]) * mask
                z = masked_instance_norm(z,mask)
                z = F.relu(z)
                z = conv1T(z, self.K[i]) * mask
                x = x - self.h*z

            # Change number of channels/resolution
            else:
                # Store the features
                xS.append(x)

                z = conv1(x, self.K[i]) * mask
                z = masked_instance_norm(z,mask)
                x = F.relu(z)

                # Downsample by factor of 2
                mS.append(mask)
                mask = F.avg_pool1d(mask.float(), 3, stride=2, padding=1).ge(0.5).long()
                x = F.avg_pool1d(x, 3, stride=2, padding=1) * mask

        # Number of scales being computed (how many downsampling)
        n_scales = len(xS)

        # Step back through the layers (up cycle)
        for i in reversed(range(1, nL)):

            # First case - Residual blocks
            # (same number of input and output kernels)
            sK = self.K[i].shape
            if sK[0] == sK[1]:
                z = conv1T(x, self.K[i]) * mask
                z = masked_instance_norm(z,mask)
                z = F.relu(z)
                z = conv1(z, self.K[i]) * mask
                x = x - self.h*z

            # Change number of channels/resolution
            else:
                n_scales -= 1
                # Upsample by factor of 2
                mask = mS[n_scales]
                x = F.interpolate(x, scale_factor=2) * mask

                z = conv1T(x, self.K[i]) * mask
                z = masked_instance_norm(z,mask)
                x = F.relu(z) + xS[n_scales]

        x = conv1(x, self.W) * mask
        D = tr2DistSmall(x)
        return (D,)

def masked_instance_norm(x, mask, eps = 1e-5):
    # ins_norm = F.instance_norm(x)
    mean = torch.sum(x * mask, dim=2) / torch.sum(mask, dim=2)
    # mean = mean.detach()
    mean_reshaped = mean.unsqueeze(2).expand_as(x)  # (N, L, C)
    var_term = ((x - mean_reshaped) * mask)**2  # (N,L,C)
    var = (torch.sum(var_term, dim=2) / torch.sum(mask, dim=2))  #(N,C)
    # var = var.detach()
    var_reshaped = var.unsqueeze(2).expand_as(x)    # (N, L, C)
    ins_norm = (x - mean_reshaped * mask) / torch.sqrt(var_reshaped + eps)   # (N, L, C)
    return ins_norm