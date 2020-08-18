import time

import matplotlib
import numpy as np

matplotlib.use('Agg')

import torch

def train(net,optimizer,dataloader_train,loss_fnc,LOG,device='cpu',epochs = 100):
    '''
    Standard training routine.
    :param net: Network to train
    :param optimizer: Optimizer to use
    :param dataloader_train: Data to train on
    :param loss_fnc: loss function to use
    :param LOG: LOG file handler to print to
    :param device: device to perform computation on
    :param dataloader_test: Dataloader to test the accuracy on after each epoch.
    :param epochs: Number of epochs to train
    :return:
    '''
    net.to(device)
    t0 = time.time()
    accuracy = 0
    for epoch in range(epochs):
        loss_train = 0
        samples_train = 0
        correct = 0
        total = 0
        for i,(seq, target) in enumerate(dataloader_train):
            if len(target) == 4:
                dist,omega,phi,theta = target[0],target[1],target[2],target[3]
            seq = seq.to(device)
            dist = dist.to(device, non_blocking=True)
            omega = omega.to(device, non_blocking=True)
            phi = phi.to(device, non_blocking=True)
            theta = theta.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = net(seq)
            loss = loss_fnc(outputs, target)
            loss.backward()
            optimizer.step()
            loss_train += loss
        t1 = time.time()
        LOG.info('{:3d}  Loss(training): {:6.2f}  LossTime: {:.2f}'.format(epoch + 1, loss_train,  t1 - t0))
    return net


