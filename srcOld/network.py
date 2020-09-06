import torch
import torch.nn as nn
import numpy as np
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
