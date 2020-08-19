import time

import matplotlib
import numpy as np

matplotlib.use('Agg')

import torch
import torch.nn as nn

class CrossEntropyMultiTargets(nn.Module):
    def __init__(self):
        super(CrossEntropyMultiTargets, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, inputs,targets,mask=None):
        loss = 0
        for (input,target) in zip(inputs,targets):
            loss += self.loss(input[:,:,mask,mask],target)
        return loss


