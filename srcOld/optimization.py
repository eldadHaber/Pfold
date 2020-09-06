import time

import matplotlib
import numpy as np

from srcOld.utils import move_tuple_to
from srcOld.visualization import compare_distogram
# from torch_lr_finder import LRFinder

matplotlib.use('Agg')

import torch

def train(net,optimizer,dataloader_train,loss_fnc,LOG,device='cpu',dl_test=None,ite=0,max_iter=100000,report_iter=1e4,checkpoint=1e19, scheduler=None):
    '''
    Standard training routine.
    :param net: Network to train
    :param optimizer: Optimizer to use
    :param dataloader_train: data to train on
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
    loss = 0
    batch =1
    # lr_finder = LRFinder(net, optimizer, loss_fnc, device=device)
    # lr_finder.range_test(dataloader_train, end_lr=100, num_iter=100)
    # lr_finder.plot()  # to inspect the loss-learning rate graph
    # lr_finder.reset()  # to reset the model and optimizer to their initial state




    while True:
        for i,(seq, target,mask) in enumerate(dataloader_train):
            seq = seq.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = net(seq,mask)
            loss += loss_fnc(outputs, target)

            loss.backward()
            optimizer.step()
            loss_train += loss.cpu().detach()
            scheduler.step()
            loss = 0

            if (ite + 1) % report_iter == 0:
                if dl_test is not None:
                    t2 = time.time()
                    loss_v = eval_net(net, dl_test, loss_fnc, device=device)
                    t3 = time.time()
                    LOG.info(
                        '{:6d}/{:6d}  Loss(training): {:6.2f}   Loss(test): {:6.2f}  LR: {:.8}  Time(train): {:.2f}  Time(test): {:.2f}  Time(total): {:.2f}  ETA: {:.2f}h'.format(
                            ite + 1,int(max_iter), loss_train/(report_iter*batch_size), loss_v/len(dl_test), scheduler.get_last_lr()[0], t2-t1, t3 - t2, t3 - t0,(max_iter-ite+1)/(ite+1)*(t3-t0)/3600))
                    t1 = time.time()
                    loss_train = 0
            if (ite + 1) % checkpoint == 0:
                pass
                # filename = "{}{}_checkpoint.tar".format(save, ite + 1)
                # save_checkpoint(ite + 1, net.state_dict(), optimizer.state_dict(), filename=filename)
                # LOG.info("Checkpoint saved: {}".format(filename))
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
    :param dataloader_train: data to train on
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
        for i,(seq, target) in enumerate(dl):
            seq = seq.to(device, non_blocking=True)
            target = move_tuple_to(target, device)

            output = net(seq)
            loss = loss_fnc(output, target)
            loss_v += loss.cpu().detach()
    compare_distogram(output, target)

    net.train()
    return loss_v

