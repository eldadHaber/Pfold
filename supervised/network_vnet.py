import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from supervised.network_transformer import tr2DistSmall, tr2DistSmall_with_std, tr2Dist_new
from supervised.visualization import plotsingleprotein, plot_coordcomparison


def conv2(X, Kernel):
    return F.conv2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def conv1(X, Kernel):
    return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))
def conv1T(X, Kernel):
    return F.conv_transpose1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def conv2T(X, Kernel):
    return F.conv_transpose2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def distConstraint(X,dc=3.79, M_fixed=None, M_padding=None):
    #TODO FIX THIS SO ITS A CLASS AND HAVE DC IN IT
    X = X.squeeze()
    n = X.shape[-1]
    nb = X.shape[0]
    if M_padding is None:
        M_padding = torch.ones_like(X[:,0,:])
    if M_fixed is None:
        M_fixed = torch.zeros_like(X[:,0,:])
    dX = X[:,:,1:] - X[:,:,:-1]
    d = torch.sum(dX**2,dim=1)
    M_padding = M_padding.squeeze()
    M_fixed = M_fixed.squeeze()

    avM = torch.round((M_padding[:,1:]+M_padding[:,:-1])/2.0) < 0.5
    dc = (avM==0)*dc
    dX = (dX / torch.sqrt(d[:,None,:]+avM[:,None,:])) * dc[:,None,:]

    Xleft = torch.zeros_like(X)
    Xleft[:,:, 0]  = X[:,:, 0]
    Xright = torch.zeros_like(X)
    Xright[:,:,-1] = X[:,:, -1]
    for i in range(1,n):
        Xleft[:,:,i] =M_fixed[:,i][:,None] * X[:,:,i] + (M_fixed[:,i] == 0).float()[:,None] * (Xleft[:,:,i-1] + dX[:,:,i-1])
        j = n-i-1
        Xright[:,:,j] = M_fixed[:,j][:,None] * X[:,:,j] + (M_fixed[:,j] == 0).float()[:,None] * (Xright[:,:,j+1] - dX[:,:,j])
    M_free = (M_fixed == 0).float()
    w = torch.zeros((nb,n,2),device=X.device)
    w[:,0,0] = M_free[:,0] * 1e10
    w[:,-1,1] = M_free[:,-1] * 1e10
    # w[:,0,0] = 0
    # w[:,-1,1] = 0

    for i in range(1,n):
        w[:,i,0] = (w[:,i-1,0] + 1) * M_free[:,i]
        j = n-i-1
        w[:,j,1] = (w[:,j+1,1] + 1) * M_free[:,j]

    # m = w > 88888
    # w[m] = 0

    wsum = torch.sum(w, dim=2,keepdim=True)
    w = (wsum-w) / wsum
    m = torch.isnan(w)
    w[m] = 0.5


    # w = w / torch.sum(w,dim=2,keepdim=True)
    w = w[:,None,:,:]
    X2 = Xleft * w[:,:,:,0] + Xright * w[:,:,:,1]

    X2 = M_padding[:,None,:]*X2

    # plot_coordcomparison(X.cpu().detach(), X2.cpu().detach(),M_fixed.cpu().detach(),num=1,plot_results=True)
    #
    dX2 = X2[:,:, 1:] - X2[:,:, :-1]
    d2 = torch.sum(dX2 ** 2, dim=1)

    torch.mean(torch.sqrt(d2))
    torch.mean(torch.sqrt(d))



    return X2





def distConstraint_fast(X,dc=3.79, M_fixed=None, M_padding=None):
    def getleftcoords(M_diff_fixed,Xleft,dX,n_individual):
        nb = Xleft.shape[0]
        for i in range(nb):
            startpoints = torch.where(M_diff_fixed[i, ...] == -1)[1]
            endpoints = torch.where(M_diff_fixed[i, ...] == 1)[1]
            if len(startpoints) == 0:
                continue
            if len(endpoints) > 0 and endpoints[0] < startpoints[0]:
                endpoints = endpoints[1:]
            if len(endpoints) < len(startpoints):
                endpoints2 = torch.empty_like(startpoints)
                endpoints2[:-1] = endpoints
                endpoints2[-1] = n_individual[i] - 1
            else:
                endpoints2 = endpoints
            for lstart, lend in zip(startpoints, endpoints2):
                a = Xleft[i, :, lstart]
                Xleft[i, :, lstart + 1:lend + 1] = a[:, None] + torch.cumsum(dX[i, :, lstart:lend], dim=-1)
        return Xleft

    def getrightcoords(M_diff_fixed,Xright,dX):
        nb,_,n = Xright.shape
        for i in range(nb):
            startpoints = torch.where(M_diff_fixed[i, ...] == 1)[1]
            endpoints = torch.where(M_diff_fixed[i, ...] == -1)[1]
            if len(startpoints) == 0:
                continue
            if len(endpoints) > 0 and endpoints[0] < startpoints[0]:
                endpoints = endpoints[1:]
            if len(endpoints) < len(startpoints):
                endpoints2 = torch.empty_like(startpoints)
                endpoints2[:-1] = endpoints
                endpoints2[-1] = n-1
            else:
                endpoints2 = endpoints
            for start, end in zip(startpoints, endpoints2):
                a = Xright[i, :, start]
                Xright[i, :, start + 1:end + 1] = a[:, None] - torch.cumsum(dX[i, :, start:end], dim=-1)
        return Xright.flip(dims=(-1,))


    #TODO FIX THIS SO ITS A CLASS AND HAVE DC IN IT
    nb,_,n = X.shape
    if M_padding is None:
        M_padding = torch.ones((nb,1,n),dtype=X.dtype,device=X.device)
    if M_fixed is None:
        M_fixed = torch.ones((nb,1,n),dtype=X.dtype,device=X.device)
    dX = X[:,:,1:] - X[:,:,:-1]
    d = torch.sum(dX**2,dim=1,keepdim=True)
    M_free = (M_fixed == 0).float()

    n_individual = torch.sum(M_padding,dim=(1,2))
    avM = torch.floor((M_padding[...,1:]+M_padding[...,:-1])/2.0) < 0.5
    dc = (avM == 0) * dc
    dX = (dX / torch.sqrt(d + avM )) * dc

    dX_flipped = torch.flip(dX,dims=(2,))


    # Xleft2 = torch.zeros_like(X)
    # Xleft2[:,:, 0]  = X[:,:, 0]
    # Xright2 = torch.zeros_like(X)
    # Xright2[:,:,-1] = X[:,:, -1]
    # for i in range(1,n):
    #     Xleft2[:,:,i] =M_fixed[...,i] * X[:,:,i] + M_free[...,i] * (Xleft2[:,:,i-1] + dX[:,:,i-1])
    #     j = n-i-1
    #     Xright2[:,:,j] = M_fixed[...,j] * X[:,:,j] + M_free[...,j] * (Xright2[:,:,j+1] - dX[:,:,j])



    M_diff_fixed = M_fixed[...,1:] - M_fixed[...,:-1]
    Xleft = X.clone()
    Xright = X.clone().flip(dims=(-1,))

    Xleft = getleftcoords(M_diff_fixed, Xleft, dX, n_individual)

    M_diff_fixed_flipped = M_diff_fixed.flip(dims=(-1,))
    Xright = getrightcoords(M_diff_fixed_flipped, Xright, dX_flipped)

    w = torch.zeros((nb,n,2),device=X.device)
    w[:,0,0] = M_free[:,0,0] * 1e10
    w[:,-1,1] = M_free[:,0,-1] * 1e10

    for i in range(1,n):
        w[:,i,0] = (w[:,i-1,0] + 1) * M_free[:,0,i]
        j = n-i-1
        w[:,j,1] = (w[:,j+1,1] + 1) * M_free[:,0,j]

    wsum = torch.sum(w, dim=2,keepdim=True)
    w = (wsum-w) / wsum
    m = torch.isnan(w)
    w[m] = 0.5

    w = w[:,None,:,:]
    #
    # wold = w.clone()
    # zeros = torch.zeros((nb,1,n),device=X.device)
    # ones = torch.ones((nb,1,n),device=X.device)
    # wl = getleftcoords(M_diff_fixed, zeros, ones, n_individual)
    # wr = getrightcoords(M_diff_fixed_flipped, zeros, -ones)
    # w = torch.empty((nb,n,2),device=X.device)
    # w[:,:,0] = wl[:,0,:]
    # w[:,:,1] = wr[:,0,:]
    # wsum = torch.sum(w, dim=2,keepdim=True)
    # w = (wsum-w) / wsum
    # m = torch.isnan(w)
    # w[m] = 0.5
    # w = w[:,None,:,:]


    X2 = Xleft * w[:,:,:,0] + Xright * w[:,:,:,1]

    X2 = M_padding*X2
    return X2





class vnet1D_inpaint(nn.Module):
    """ VNet """
    def __init__(self, chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size,cross_dist=False, dropout_p=0.0):
        super(vnet1D_inpaint, self).__init__()
        K, W = self.init_weights(chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size=stencil_size)
        self.K = K
        self.W = W
        self.h = 0.1
        self.cross_dist = cross_dist
        self.dropout = nn.Dropout(dropout_p)

    def init_weights(self, chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size = 3):
        K = nn.ParameterList([])
        for i in range(nblocks):
            # First the channel change.
            if i == 0:
                chan_i = chan_in
            else:
                chan_i = channels * 2**(i - 1)
            chan_o = channels * 2**i
            stdv = 1e-1 * chan_i / chan_o
            Ki = torch.zeros(chan_o, chan_i, stencil_size)
            Ki.data.uniform_(-stdv, stdv)
            Ki = nn.Parameter(Ki)
            K.append(Ki)

            if i != nblocks-1:
                # Last block is just a coarsening, since it is a vnet and not a unet
                for j in range(nlayers_pr_block):
                    stdv = 1e-2
                    chan = channels * 2 ** i
                    Ki = torch.zeros(chan, chan, stencil_size)
                    Ki.data.uniform_(-stdv, stdv)
                    Ki = nn.Parameter(Ki)
                    K.append(Ki)

        W = nn.Parameter(1e-1*torch.randn(chan_out, channels, 1))
        return K, W

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x[:,0,:])

        coords = x[:,-4:-1,:]
        tmp = x[:,-1,:] == 1
        mask_coords = torch.repeat_interleave(tmp[:,None,:],3,dim=1)
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
        x = self.dropout(x)

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

        #Now we anchor in the coordinates that we know
        x[mask_coords] = coords[mask_coords]

        #next we constrain the coordinates in the predicted to have an acceptable range
        # t1 = time.time()
        # x2 = distConstraint(x,M_fixed=mask_coords[:,:1,:].float(),M_padding=mask)
        # t2 = time.time()
        x = distConstraint_fast(x,M_fixed=mask_coords[:,:1,:].float(),M_padding=mask)
        # t3 = time.time()
        # print("time {:2.4f}, time {:2.4f}".format(t2-t1,t3-t2))

        if self.cross_dist:
            nl = x.shape[-1]
            nc = x.shape[1]//3
            x_long = torch.empty_like(x)
            x_long = x_long.reshape((x.shape[0],3,-1))
            for i in range(x.shape[1]//3):
                x_long[:,:,i*nl:(i+1)*nl] = x[:,i*3:(i+1)*3,:]
            dist_long = tr2DistSmall(x_long)
            dists = ()
            for i in range(nc):
                for j in range(nc):
                    dists += (dist_long[:,i*nl:(i+1)*nl,j*nl:(j+1)*nl],)



        else:
            dists = ()
            for i in range(x.shape[1]//3):
                dists += (tr2Dist_new(x[:,i*3:(i+1)*3,:]),)

        return dists, x





class vnet1D(nn.Module):
    """ VNet """
    def __init__(self, chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size,cross_dist=False, dropout_p=0.0):
        super(vnet1D, self).__init__()
        K, W = self.init_weights(chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size=stencil_size)
        self.K = K
        self.W = W
        self.h = 0.1
        self.cross_dist = cross_dist
        self.dropout = nn.Dropout(dropout_p)

    def init_weights(self, chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size = 3):
        K = nn.ParameterList([])
        for i in range(nblocks):
            # First the channel change.
            if i == 0:
                chan_i = chan_in
            else:
                chan_i = channels * 2**(i - 1)
            chan_o = channels * 2**i
            stdv = 1e-1 * chan_i / chan_o
            Ki = torch.zeros(chan_o, chan_i, stencil_size)
            Ki.data.uniform_(-stdv, stdv)
            Ki = nn.Parameter(Ki)
            K.append(Ki)

            if i != nblocks-1:
                # Last block is just a coarsening, since it is a vnet and not a unet
                for j in range(nlayers_pr_block):
                    stdv = 1e-2
                    chan = channels * 2 ** i
                    Ki = torch.zeros(chan, chan, stencil_size)
                    Ki.data.uniform_(-stdv, stdv)
                    Ki = nn.Parameter(Ki)
                    K.append(Ki)

        W = nn.Parameter(1e-1*torch.randn(chan_out, channels, 1))
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
        x = self.dropout(x)
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
        if self.cross_dist:
            nl = x.shape[-1]
            nc = x.shape[1]//3
            x_long = torch.empty_like(x)
            x_long = x_long.reshape((x.shape[0],3,-1))
            for i in range(x.shape[1]//3):
                x_long[:,:,i*nl:(i+1)*nl] = x[:,i*3:(i+1)*3,:]
            dist_long = tr2DistSmall(x_long)
            dists = ()
            for i in range(nc):
                for j in range(nc):
                    dists += (dist_long[:,i*nl:(i+1)*nl,j*nl:(j+1)*nl],)



        else:
            dists = ()
            for i in range(x.shape[1]//3):
                dists += (tr2DistSmall(x[:,i*3:(i+1)*3,:]),)

        return dists, x

    def NNreg(self):
        RW  = torch.norm(self.W)**2/2/self.W.numel()
        RK = 0
        # for i in range(len(self.K)-1): #TV REG
        #     dKdt = self.K[i+1] - self.K[i]
        #     RK += torch.sum(torch.abs(dKdt))/dKdt.numel()
        for i in range(len(self.K)):
            RK += torch.norm(self.K[i])**2/2/self.K[i].numel()
        return RW + RK


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
