import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def conv1(X, Kernel):
    return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))
def conv1T(X, Kernel):
    return F.conv_transpose1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def tv_norm(X,epsilon=1e-2):
    #X = X - torch.mean(X,dim=1,keepdim=True)
    #X = X/torch.sqrt(torch.mean(X**2,dim=1,keepdim=True) + epsilon)
    X = F.instance_norm(X)
    return X

##### 1D VNET ###########################
#
#
#
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

    def forward(self, x, m=1.0):
        """ Forward propagation through the network """

        # Number of layers
        nL = len(self.K)
        # Store the output at different scales to add back later
        xS = []
        mS = [m]
        # Opening layer
        z = conv1(x, self.K[0])
        z = tv_norm(z)
        x = F.relu(z)

        # Step through the layers (down cycle)
        for i in range(1, nL):

            # First case - Residual blocks
            # (same number of input and output kernels)

            sK = self.K[i].shape

            if sK[0] == sK[1]:
                z  = conv1(x, self.K[i])
                z  = tv_norm(z)
                z  = F.relu(z)
                z  = conv1T(z, self.K[i])
                x  = x - self.h*z

            # Change number of channels/resolution
            else:
                # Store the features
                xS.append(x)
                z  = conv1(x, self.K[i])
                z  = tv_norm(z)
                x  = F.relu(z)

                # Downsample by factor of 2
                x = F.avg_pool1d(x, 3, stride=2, padding=1)
                m = F.avg_pool1d(m.unsqueeze(0).unsqueeze(0), 3, stride=2, padding=1).squeeze(0).squeeze(0)
                mS.append(m)
        # Number of scales being computed (how many downsampling)
        n_scales = len(xS)

        # Step back through the layers (up cycle)
        m = mS[-1]
        for i in reversed(range(1, nL)):
            # First case - Residual blocks
            # (same number of input and output kernels)
            sK = self.K[i].shape
            if sK[0] == sK[1]:
                z  = conv1T(x, self.K[i])
                z  = tv_norm(z)
                z  = F.relu(z)
                z  = conv1(z, self.K[i])
                x  = x - self.h*z

            # Change number of channels/resolution
            else:
                n_scales -= 1
                # Upsample by factor of 2
                x = F.interpolate(x, scale_factor=2)
                mS = mS[:-1]
                m  = mS[-1]
                z  = conv1T(x, self.K[i])
                z  = tv_norm(z)
                x  = F.relu(z) + xS[n_scales]

        x = conv1(x, self.W)
        return x


class proDis(nn.Module):
    """ protein discreminator """
    def __init__(self, Arch,nout):
        super(proDis, self).__init__()
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

        W = nn.Parameter(1e-4*torch.randn(nout, A[-1,1], 1))
        npar += W.numel()
        print('Number of parameters  ', npar)
        return K, W

    def forward(self, x):
        """ Forward propagation through the network """

        # Number of layers
        nL = len(self.K)
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
                z  = tv_norm(z)
                z  = F.relu(z)
                z  = conv1T(z, self.K[i])
                x  = x - self.h*z
            # Change number of channels/resolution
            else:
                z  = conv1(x, self.K[i])
                z  = tv_norm(z)
                x  = F.relu(z)

                # Downsample by factor of 2
                x = F.avg_pool1d(x, 3, stride=2, padding=1)

        return conv1(x, self.W)

class proGen(nn.Module):
    """ VNet """
    def __init__(self, Arch,nout):
        super(proGen, self).__init__()
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

    def forward(self, x, m=1.0):
        """ Forward propagation through the network """

        # Number of layers
        nL = len(self.K)

        # Store the output at different scales to add back later
        #xS = []
        mS = [m]
        # Opening layer
        z = conv1(x, self.K[0])
        z = tv_norm(z)
        x = F.relu(z)

        # Step through the layers (down cycle)
        for i in range(1, nL):
            # First case - Residual blocks
            # (same number of input and output kernels)
            sK = self.K[i].shape

            if sK[0] == sK[1]:
                z  = conv1(x, self.K[i])
                z  = tv_norm(z)
                z  = F.relu(z)
                z  = conv1T(z, self.K[i])
                x  = x - self.h*z

            # Change number of channels/resolution
            else:
                # Store the features
                #xS.append(x)
                z  = conv1(x, self.K[i])
                z  = tv_norm(z)
                x  = F.relu(z)

                # Downsample by factor of 2
                x = F.avg_pool1d(x, 3, stride=2, padding=1)
                m = F.avg_pool1d(m.unsqueeze(0).unsqueeze(0), 3, stride=2, padding=1).squeeze(0).squeeze(0)
                mS.append(m)
        # Number of scales being computed (how many downsampling)
        n_scales = len(mS)

        # Step back through the layers (up cycle)
        for i in reversed(range(1, nL)):

            # First case - Residual blocks
            # (same number of input and output kernels)
            sK = self.K[i].shape
            if sK[0] == sK[1]:
                z  = conv1T(x, self.K[i])
                z  = tv_norm(z)
                z  = F.relu(z)
                z  = conv1(z, self.K[i])
                x  = x - self.h*z

            # Change number of channels/resolution
            else:
                n_scales -= 1
                # Upsample by factor of 2
                x = F.interpolate(x, scale_factor=2)
                mS = mS[:-1]
                z  = conv1T(x, self.K[i])
                z  = tv_norm(z)
                x  = F.relu(z) # + xS[n_scales]

        x = conv1(x, self.W)
        return x



class generator(nn.Module):

    # generator model
    def __init__(self):
        super(generator, self).__init__()

        self.t1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=(3), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=(3), stride=2, padding=1),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=(3), stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=(3), stride=2, padding=1),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=(3), stride=2, padding=1),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t6 = nn.Sequential(
            nn.Conv1d(512, 4000, kernel_size=(1)),
            # bottleneck
            nn.InstanceNorm1d(4000),
            nn.ReLU(),
            nn.Conv1d(4000, 512, kernel_size=(1)),
        )

        self.t7 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=(4), stride=2, padding=1),
            nn.InstanceNorm1d(256),
            nn.ReLU()
        )
        self.t8 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=(4), stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU()
        )
        self.t9 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=(4), stride=2, padding=1),
            nn.InstanceNorm1d(64),
            nn.ReLU()
        )
        self.t10 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=3, kernel_size=(4), stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)
        x = self.t5(x)
        x = self.t6(x)
        x = self.t7(x)
        x = self.t8(x)
        x = self.t9(x)
        x = self.t10(x)
        x = torch.max_pool1d(x,4,padding=1)
        return x  # output of generator


import torch
from torch import nn


class discriminator(nn.Module):

    # discriminator model
    def __init__(self):
        super(discriminator, self).__init__()

        self.t1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=(3), stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=(3), stride=1, padding=1),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=(3), stride=1, padding=1),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t5 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=(1), stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.t1(x)
        x = torch.max_pool1d(x, 2, padding=0)
        x = self.t2(x)
        x = torch.max_pool1d(x, 2, padding=0)
        x = self.t3(x)
        x = torch.max_pool1d(x, 2, padding=0)
        x = self.t4(x)
        x = torch.max_pool1d(x, 2, padding=0)
        x = self.t5(x)
        return 0.5*(torch.tanh(torch.sum(x))+1)  # output of discriminator