import time

import matplotlib
import numpy as np

matplotlib.use('Agg')

import torch
import torch.nn as nn

class LossMultiTargets(nn.Module):
    def __init__(self,loss_fnc=torch.nn.CrossEntropyLoss()):
        super(LossMultiTargets, self).__init__()
        self.loss = loss_fnc

    def forward(self, inputs,targets):
        # loss = []
        # for (input,target) in zip(inputs,targets):
        #     loss.append(self.loss(input,target))
        loss = 0
        for (input,target) in zip(inputs,targets):
            loss += self.loss(input,target)
        return loss

class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss,self).__init__()

    def forward(self, input, target):
        #We only want places where the target is larger than zero (remember this is for distances)
        mask = target > 0
        # result = torch.mean((input[mask] - target[mask])**2)
        # result = torch.norm((input[mask] - target[mask])) ** 2 / torch.norm(target[mask]) ** 2
        nb = target.shape[0]
        result = 0
        for i in range(nb):
            inputi = input[i,:,:]
            targeti = target[i,:,:]
            maski = targeti > 0
            assert torch.norm(targeti[maski]) > 0
            result += torch.norm((inputi[maski] - targeti[maski])) ** 2 / torch.norm(targeti[maski]) ** 2

        return result/nb




