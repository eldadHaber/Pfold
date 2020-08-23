import time

import matplotlib
import numpy as np

from src.utils import move_tuple_to
from src.visualization import compare_distogram

matplotlib.use('Agg')

import torch

def train(net,optimizer,dataloader_train,loss_fnc,LOG,device='cpu',dl_test=None,ite=0,max_iter=100000,report_iter=1e4,checkpoint=1e19):
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
    stop_run = False
    net.to(device)
    t0 = time.time()
    t1 = time.time()
    loss_train = 0
    while True:
        losses_val = [0, 0, 0, 0]
        for i,(seq, target, mask) in enumerate(dataloader_train):
            seq = seq.to(device)
            target = move_tuple_to(target, device)
            optimizer.zero_grad()
            outputs = net(seq)
            losses = loss_fnc(outputs, target)
            for i,loss in enumerate(losses):
                losses_val[i] += loss
            loss_sum = sum(losses)
            loss_sum.backward()
            optimizer.step()
            loss_train += loss_sum
            if (ite + 1) % report_iter == 0:
                if dl_test is not None:
                    t2 = time.time()
                    loss_v = eval_net(net, dl_test, loss_fnc, device=device)
                    t3 = time.time()
                    LOG.info(
                        '{:6d}/{:6d}  Loss(training): {:6.2f}   Loss(1): {:6.2f}   Loss(2): {:6.2f}   Loss(3): {:6.2f}   Loss(4): {:6.2f}   Loss(test): {:6.2f}  Time(train): {:.2f}  Time(test): {:.2f}  Time(total): {:.2f}'.format(
                            ite + 1,int(max_iter), loss_train, losses_val[0], losses_val[1], losses_val[2], losses_val[3], loss_v, t2-t1, t3 - t2, t3 - t0))
                    t1 = time.time()
                    loss_train = 0
            if (ite + 1) % checkpoint == 0:
                filename = "{}{}_checkpoint.tar".format(save, ite + 1)
                save_checkpoint(ite + 1, net.state_dict(), optimizer.state_dict(), filename=filename)
                LOG.info("Checkpoint saved: {}".format(filename))
            ite += 1
            if ite >= max_iter:
                stop_run = True
                break
        if stop_run:
            break
    return net


def eval_net(net, dl, loss_fnc, device='cpu'):
    '''
    Standard training routine.
    :param net: Network to train
    :param optimizer: Optimizer to use
    :param dataloader_train: Data to train on
    :param loss_fnc: loss function to use
    :param device: device to perform computation on
    :param dataloader_test: Dataloader to test the accuracy on after each epoch.
    :param epochs: Number of epochs to train
    :return:
    '''
    net.to(device)
    net.eval()
    with torch.no_grad():
        loss_v = 0
        for i,(seq, target, mask)  in enumerate(dl):
            seq = seq.to(device)
            target = move_tuple_to(target, device)

            output = net(seq)
            losses = loss_fnc(output, target)
            loss_v += sum(losses)
    compare_distogram(output, target)

    net.train()
    return loss_v

