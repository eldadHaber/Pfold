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




#
# def distConstraint(X,dc=3.79, M_fixed=None, M_padding=None):
#     #TODO FIX THIS SO ITS A CLASS AND HAVE DC IN IT
#     X = X.squeeze()
#     n = X.shape[-1]
#     nb = X.shape[0]
#     if M_padding is None:
#         M_padding = torch.ones_like(X[:,0,:])
#     if M_fixed is None:
#         M_fixed = torch.zeros_like(X[:,0,:])
#     dX = X[:,:,1:] - X[:,:,:-1]
#     d = torch.sum(dX**2,dim=1)
#
#     avM = torch.round((M_padding[:,1:]+M_padding[:,:-1])/2.0) < 0.5
#     dc = (avM==0)*dc
#     dX = (dX / torch.sqrt(d[:,None,:]+avM[:,None,:])) * dc[:,None,:]
#
#     Xleft = torch.zeros_like(X)
#     Xleft[:,:, 0]  = X[:,:, 0]
#     Xright = torch.zeros_like(X)
#     Xright[:,:,-1] = X[:,:, -1]
#     for i in range(1,n):
#         Xleft[:,:,i] =M_fixed[:,i][:,None] * X[:,:,i] + (M_fixed[:,i] == 0).float()[:,None] * (Xleft[:,:,i-1] + dX[:,:,i-1])
#         j = n-i-1
#         Xright[:,:,j] = M_fixed[:,j][:,None] * X[:,:,j] + (M_fixed[:,j] == 0).float()[:,None] * (Xright[:,:,j+1] - dX[:,:,j])
#     w = torch.zeros((nb,n,2),device=X.device)
#     w[:,0,0] = M_fixed[:,0]
#     w[:,-1,1] = M_fixed[:,-1]
#
#     for i in range(1,n):
#         w[:,i,0] = torch.max(M_fixed[:,i],w[:,i-1,0])
#         j = n-i-1
#         w[:,j,1] = torch.max(M_fixed[:,j],w[:,j+1,1])
#
#     w = w / torch.sum(w,dim=2,keepdim=True)
#     w = w[:,None,:,:]
#     X2 = Xleft * w[:,:,:,0] + Xright * w[:,:,:,1]
#
#     X2 = M_padding[:,None,:]*X2
#
#     # plot_coordcomparison(X.cpu().detach(), X2.cpu().detach(),M_fixed.cpu().detach(),num=1,plot_results=True)
#     #
#     dX2 = X2[:,:, 1:] - X2[:,:, :-1]
#     d2 = torch.sum(dX2 ** 2, dim=1)
#
#     torch.mean(torch.sqrt(d2))
#     torch.mean(torch.sqrt(d))
#
#
#
#     return X2
#
# #


#
# n=9
# X = torch.zeros((3,n),dtype=torch.float32)
# X2 = torch.zeros((2,3,n),dtype=torch.float32)
# X[0,1] = 4
# X[0,2] = 7
# X[0,3] = 9
# X[0,4] = 12
# X[0,5] = 13
# X[0,6] = 15
# X[0,7] = 16
# X[0,8] = 21
#
# X[1,1] = 1
# X[1,2] = 2
# X[1,3] = 2
# X[1,4] = 3
# X[1,5] = 4
# X[1,6] = 3
# X[1,7] = 2
# X[1,8] = 1
#
# #
# M_f = torch.zeros_like(X[0,:])
# M_f[2] = 1
# M_f[3] = 1
# M_f[6] = 1
#
# M2 = torch.zeros((2,n),dtype=torch.float32)
# M2[0,:] = M_f
# M2[1,:] = M_f
# #
# X2[0,:,:] = X
# X2[1,:,:] = X
# distConstraint(X2,dc=2, M_fixed=M2, M_padding=None)
#
# print("test")

class vnet1D_inpaint(nn.Module):
    """ VNet """
    def __init__(self, chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size,cross_dist=False):
        super(vnet1D_inpaint, self).__init__()
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
        x = distConstraint(x,M_fixed=mask_coords[:,0,:].float(),M_padding=mask[:,0,:])

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
