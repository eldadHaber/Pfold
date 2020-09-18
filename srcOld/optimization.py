import time

import matplotlib
import numpy as np

from srcOld.loss import loss_tr, loss_tr_all, loss_tr_tuples
from srcOld.utils import move_tuple_to
from srcOld.visualization import compare_distogram, plotcoordinates, plotfullprotein

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
    loss_train_d = 0
    loss_train_ot = 0
    loss = 0
    enable_coordinate_loss = False
    tt0 = time.time()
    while True:
        for i,(seq, dists,mask, coords) in enumerate(dataloader_train):

            tt1 = time.time()
            seq = seq.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True) # Note that this is the padding mask, and not the mask for targets that are not available.
            dists = move_tuple_to(dists, device, non_blocking=True)
            coords = move_tuple_to(coords, device, non_blocking=True)
            optimizer.zero_grad()
            tt11 = time.time()
            dists_pred, coords_pred = net(seq,mask)
            tt12 = time.time()

            loss_c = loss_tr_tuples(coords_pred, coords)
            tc = time.time()
            loss_d = loss_fnc(dists_pred, dists)
            td = time.time()

            loss = 0.5 * loss_d + 0.5 * loss_c
            tt13 = time.time()
            loss.backward()
            optimizer.step()
            tt14 = time.time()
            loss_train_d += loss_d.cpu().detach()
            loss_train_ot += loss_c.cpu().detach()
            if scheduler is not None:
                scheduler.step()

            tt2 = time.time()
            # print("Loading:{:2.4f}, other:{:2.4f}, transfer:{:2.4f}, network:{:2.4f}, loss:{:2.4f}, backward:{:2.4f}, loss_c:{:2.4f}, loss_d:{:2.4f}".format(tt1-tt0,tt2-tt1,tt11-tt1,tt12-tt11,tt13-tt12,tt14-tt13,tc-tt12,td-tc))
            tt0 = time.time()
            if (ite + 1) % report_iter == 0:
                if dl_test is not None:
                    t2 = time.time()
                    loss_v = eval_net(net, dl_test, loss_fnc, device=device, plot_results=True)
                    t3 = time.time()
                    if scheduler is None:
                        lr = optimizer.param_groups[0]['lr']
                    else:
                        lr = scheduler.get_last_lr()[0]
                    LOG.info(
                        '{:6d}/{:6d}  Loss(training): {:6.4f}%  Loss(test): {:6.4f}%  Loss(dist): {:6.4f}%  Loss(coord): {:6.4f}%  LR: {:.8}  Time(train): {:.2f}  Time(test): {:.2f}  Time(total): {:.2f}  ETA: {:.2f}h'.format(
                            ite + 1,int(max_iter), (loss_train_d+loss_train_ot)/2/report_iter*100, loss_v*100, loss_train_d/2/report_iter*100, loss_train_ot/2/report_iter*100, lr, t2-t1, t3 - t2, t3 - t0,(max_iter-ite+1)/(ite+1)*(t3-t0)/3600))
                    t1 = time.time()
                    loss_train_d = 0
                    loss_train_ot = 0
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
    # plotfullprotein(cnn_pred, caa_pred, cbb_pred, cnn_target, caa_target, cbb_target)
    # plotcoordinates(pred, target_coord)
    return net


def eval_net(net, dl, loss_fnc, device='cpu', plot_results=False):
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
        for i,(seq, dists,mask, coords) in enumerate(dl):
            seq = seq.to(device, non_blocking=True)
            dists = move_tuple_to(dists, device, non_blocking=True)
            coords = move_tuple_to(coords, device, non_blocking=True)
            dists_pred, coords_pred = net(seq)
            loss_c, coords_pred_tr, coords_tr = loss_tr_tuples(coords_pred, coords, return_coords=True)
            loss_d = loss_fnc(dists_pred, dists)
            loss_v += (loss_d+loss_c).cpu().detach()/2
    if plot_results:
        compare_distogram(dists_pred, dists)
        plotfullprotein(coords_pred_tr, coords_tr)
    net.train()
    return loss_v/len(dl)



def OT(r1,r2):
    '''
    We try to do optimal transport of a point cloud into a reference point cloud.
    The transport is done with translation and rotation only.
    '''

    #r2 gives the mask
    mask = r2 != 0
    nb = r1.shape[0]

    r1s = r1[mask].reshape(nb,3,-1)
    r2s = r2[mask].reshape(nb,3,-1)

    #First we translate the two sets, by setting both their centroids to origin
    r1centroid = torch.mean(r1s,dim=2)
    r2centroid = torch.mean(r2s,dim=2)

    r1c = r1s - r1centroid[:,:,None]
    r2c = r2s - r2centroid[:,:,None]

    H = r1c @ r2c.transpose(1,2)

    #R = torch.matrix_power(H.transpose(1,2) @ H,0.5) @ torch.inverse(H)
    U, S, V = torch.svd(H)

    d = torch.det(V @ U.transpose(1,2))

    tt = torch.tensor([1, 1, d])
    tmp = torch.diag_embed(tt).to(device=V.device)
    R = V @ tmp @ U.transpose(1,2)

    tt2 = torch.tensor([1, 1, -d])
    tmp2 = torch.diag_embed(tt2).to(device=V.device)
    R2 = V @ tmp2 @ U.transpose(1,2)

    r1c_rotated = R @ r1c
    r1c_rotated2 = R2 @ r1c

    assert torch.norm(r2c) > 0
    res1 = torch.norm(r1c_rotated - r2c) ** 2 / torch.norm(r2c) ** 2
    res2 = torch.norm(r1c_rotated2 - r2c) ** 2 / torch.norm(r2c) ** 2


    # dr11 = r1c_rotated[:,:,1:] - r1c_rotated[:,:,:-1]
    # dr12 = r1c_rotated2[:,:,1:] - r1c_rotated2[:,:,:-1]
    # dr2 = r2c[:,:,1:] - r2c[:,:,:-1]
    #
    # dres1 = torch.norm(dr11 - dr2) ** 2 / torch.norm(dr2) ** 2
    # dres2 = torch.norm(dr12 - dr2) ** 2 / torch.norm(dr2) ** 2


    if res1 < res2:
        res = res1
        # dres = dres1
        pred = r1c_rotated.squeeze().cpu().detach().numpy()

    else:
        pred = r1c_rotated2.squeeze().cpu().detach().numpy()
        res = res2
        # dres = dres2
    result = res #+ dres
    # print("result = {:2.2f}, result2 = {:2.2f}".format(result,result2))

    target = r2c.squeeze().cpu().detach().numpy()

    return result, pred,target