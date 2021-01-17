import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from supervised.network_transformer import tr2DistSmall



class ResNet(nn.Module):
    def __init__(self, chan_in, chan_out, channels, nblocks, nlayers_pr_block, stencil_size=3):
        super(ResNet, self).__init__()
        self.conv_init = nn.Conv2d(chan_in, channels, kernel_size=1)
        self.norm_init = nn.InstanceNorm2d(channels)

        blocks = []
        for i in range(nblocks):
            for j in range(nlayers_pr_block):
                dilation = 2**j
                block = ResidualBlock(channels,dilation)
                blocks.append(block)
        self.blocks = nn.Sequential(*blocks)
        self.activation = nn.ELU()
        # self.dNN = nn.Conv2d(nf, 1, kernel_size=3, padding=1)
        # self.dCaCa = nn.Conv2d(nf, 1, kernel_size=3, padding=1)
        self.dCbCb = nn.Conv2d(channels, 1, kernel_size=stencil_size, padding=1)
        # self.dNCa = nn.Conv2d(nf, 1, kernel_size=3, padding=1)
        # self.dNCb = nn.Conv2d(nf, 1, kernel_size=3, padding=1)
        # self.dCaCb = nn.Conv2d(nf, 1, kernel_size=3, padding=1)

    def forward(self, x, mask):
        x = self.conv_init(x)
        x = self.norm_init(x)
        x = self.activation(x)

        for block in self.blocks:
            x = block(x)
        # dNCa = self.dNCa(x)
        # dNCb = self.dNCb(x)
        # dCaCb = self.dCaCb(x)
        y = 0.5 * (x + torch.transpose(x, -1, -2))
        # dNN = self.dNN(y)
        # dCaCa = self.dCaCa(y)
        dCbCb = self.dCbCb(y)
        return dCbCb, None
        # return dNN, dCaCa, dCbCb, dNCa, dNCb, dCaCb


class ResidualBlock(nn.Module):
    def __init__(self, nf,dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(nf,nf, kernel_size=3,dilation=dilation, padding=dilation)
        self.norm1 = nn.InstanceNorm2d(nf)
        self.dropout = nn.Dropout2d(0.15)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, dilation=dilation, padding=dilation)
        self.norm2 = nn.InstanceNorm2d(nf)
        self.activation = nn.ELU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.activation(y+x)
        return y


