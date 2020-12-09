import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.network_transformer import tr2DistSmall, tr2DistSmall_with_std


def conv2(X, Kernel):
    return F.conv2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def conv1(X, Kernel):
    return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))
def conv1T(X, Kernel):
    return F.conv_transpose1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def conv2T(X, Kernel):
    return F.conv_transpose2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))




class vnet1D(nn.Module):
    """ VNet """
    def __init__(self, chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size,cross_dist=False):
        super(vnet1D, self).__init__()
        K, W = self.init_weights(chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size=stencil_size)
        self.K = K
        self.W = W
        self.h = 0.1
        self.cross_dist = cross_dist

    def init_weights(self, chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size = 3):
        K = nn.ParameterList([])
        for i in range(nblocks):
            # First the channel change.
            if i == 0:
                chan_i = chan_in
            else:
                chan_i = channels * 2**(i - 1)
            chan_o = channels * 2**i
            stdv = 1e-2 * chan_i / chan_o
            Ki = torch.zeros(chan_o, chan_i, stencil_size)
            Ki.data.uniform_(-stdv, stdv)
            Ki = nn.Parameter(Ki)
            K.append(Ki)

            if i != nblocks-1:
                # Last block is just a coarsening, since it is a vnet and not a unet
                for j in range(nlayers_pr_block):
                    stdv = 1e-3
                    chan = channels * 2 ** i
                    Ki = torch.zeros(chan, chan, stencil_size)
                    Ki.data.uniform_(-stdv, stdv)
                    Ki = nn.Parameter(Ki)
                    K.append(Ki)

        W = nn.Parameter(1e-2*torch.randn(chan_out, channels, 1))
        return K, W

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x[:,0,:])

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

        pssm = x[:,:20,:]
        # pssm = torch.softmax(pssm,dim=1) * mask
        entropy = x[:,20:,:]
        entropy = torch.tanh(torch.abs(entropy)) * mask

        return pssm, entropy

def masked_instance_norm(x, mask, eps = 1e-5):
    # ins_norm = F.instance_norm(x)
    mean = torch.sum(x * mask, dim=2) / torch.sum(mask, dim=2)
    # mean = mean.detach()
    mean_reshaped = mean.unsqueeze(2).expand_as(x) * mask  # (N, L, C)
    var_term = ((x - mean_reshaped) * mask)**2  # (N,L,C)
    var = (torch.sum(var_term, dim=2) / torch.sum(mask, dim=2))  #(N,C)
    # var = var.detach()
    var_reshaped = var.unsqueeze(2).expand_as(x)    # (N, L, C)
    ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + eps)   # (N, L, C)
    return ins_norm




def masked_instance_norm_2d(x, mask, eps = 1e-5):
    mean = torch.sum(x * mask, dim=(-1,-2)) / torch.sum(mask, dim=(-1,-2))
    # mean = mean.detach()
    mean_reshaped = mean[:,:,None,None].expand_as(x) * mask  # (N, L, C)
    var_term = ((x - mean_reshaped) * mask)**2  # (N,L,C)
    var = (torch.sum(var_term, dim=(-1,-2)) / torch.sum(mask, dim=(-1,-2)))  #(N,C)
    # var = var.detach()
    var_reshaped = var[:,:,None,None].expand_as(x)    # (N, L, C)
    ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + eps)   # (N, L, C)
    return ins_norm




class vnet2D(nn.Module):
    """ VNet """
    def __init__(self, chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size):
        super(vnet2D, self).__init__()
        K, W = self.init_weights(chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size=stencil_size)
        self.K = K
        self.W = W
        self.h = 0.1

    def init_weights(self, chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size = 3):
        K = nn.ParameterList([])
        for i in range(nblocks):
            # First the channel change.
            if i == 0:
                chan_i = chan_in
            else:
                chan_i = channels * 2**(i - 1)
            chan_o = channels * 2**i
            stdv = 1e-2 * chan_i / chan_o

            # m = nn.Conv2d(16, 33, 3, stride=2)


            Ki = torch.zeros(chan_o, chan_i, stencil_size, stencil_size)
            Ki.data.uniform_(-stdv, stdv)

            Ki = nn.Parameter(Ki)
            K.append(Ki)

            if i != nblocks-1:
                # Last block is just a coarsening, since it is a vnet and not a unet
                for j in range(nlayers_pr_block):
                    stdv = 1e-3
                    chan = channels * 2 ** i
                    Ki = torch.zeros(chan, chan, stencil_size, stencil_size)
                    Ki.data.uniform_(-stdv, stdv)
                    Ki = nn.Parameter(Ki)
                    K.append(Ki)

        W = nn.Parameter(1e-4*torch.randn(chan_out, channels, 1, 1))
        return K, W

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x[:,0,:])

        """ Forward propagation through the network """

        mask = mask.unsqueeze(1)
        mm = mask.transpose(1,2) * mask
        mm = mm.unsqueeze(1)

        # Number of layers
        nL = len(self.K)

        # Store the output at different scales to add back later
        xS = []

        # Store the masks at different scales to mask with later
        mS = []

        # Opening layer
        z = conv2(x, self.K[0]) * mm
        z = masked_instance_norm_2d(z,mm)
        x = F.relu(z)

        # Step through the layers (down cycle)
        for i in range(1, nL):

            # First case - Residual blocks
            # (same number of input and output kernels)

            sK = self.K[i].shape

            if sK[0] == sK[1]:
                z = conv2(x, self.K[i]) * mm
                z = masked_instance_norm_2d(z,mm)
                z = F.relu(z)
                z = conv2T(z, self.K[i]) * mm
                x = x - self.h*z

            # Change number of channels/resolution
            else:
                # Store the features
                xS.append(x)

                z = conv2(x, self.K[i]) * mm
                z = masked_instance_norm_2d(z,mm)
                x = F.relu(z)

                # Downsample by factor of 2
                mS.append(mm)
                mm = F.avg_pool2d(mm.float(), 3, stride=2, padding=1).ge(0.5).long()
                x = F.avg_pool2d(x, 3, stride=2, padding=1) * mm

        # Number of scales being computed (how many downsampling)
        n_scales = len(xS)

        # Step back through the layers (up cycle)
        for i in reversed(range(1, nL)):

            # First case - Residual blocks
            # (same number of input and output kernels)
            sK = self.K[i].shape
            if sK[0] == sK[1]:
                z = conv2T(x, self.K[i]) * mm
                z = masked_instance_norm_2d(z,mm)
                z = F.relu(z)
                z = conv2(z, self.K[i]) * mm
                x = x - self.h*z

            # Change number of channels/resolution
            else:
                n_scales -= 1
                # Upsample by factor of 2
                mm = mS[n_scales]
                x = F.interpolate(x, scale_factor=2) * mm

                z = conv2T(x, self.K[i]) * mm
                z = masked_instance_norm_2d(z,mm)
                x = F.relu(z) + xS[n_scales]

        x = conv2(x, self.W) * mm

        # dists = ()
        # for i in range(x.shape[1]//3):
        #     dists += (tr2DistSmall(x[:,i:i+3,:]),)

        return x, None
