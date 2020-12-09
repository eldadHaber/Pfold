import time

import matplotlib

from src.IO import save_checkpoint
from src.loss import loss_tr_tuples, Loss_reg_min_separation, LossMultiTargets
from src.utils import move_tuple_to
from src.visualization import compare_distogram, plotfullprotein

# from torch_lr_finder import LRFinder

matplotlib.use('Agg')

import torch

def train(net,optimizer,dataloader_train,loss_fnc,LOG,device='cpu',dl_test=None,ite=0,max_iter=100000,report_iter=1e4,checkpoint=1e19, scheduler=None, sigma=-1, save=None, use_loss_coord=True, viz=False, loss_reg_fnc=None):
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
    loss_train_pssm = 0
    loss_train_entropy = 0
    loss_train = 0
    logsoft = torch.nn.LogSoftmax(dim=1)
    KLloss = torch.nn.KLDivLoss(reduction='none')
    while True:
        for i,(features, pssm, mask, entropy) in enumerate(dataloader_train):
            features = features.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True) # Note that this is the padding mask, and not the mask for targets that are not available.
            pssm = pssm.to(device, non_blocking=True)
            entropy = entropy.to(device, non_blocking=True)

            optimizer.zero_grad()
            pssm_pred, entropy_pred = net(features,mask)
            loss_pssm = torch.mean(torch.sum(KLloss(logsoft(pssm_pred),pssm),dim=(1,2)) / torch.sum(mask,dim=1))

            loss_entropy = torch.mean(torch.norm(entropy - entropy_pred,dim=1))

            loss = loss_pssm + loss_entropy

            loss.backward()
            optimizer.step()
            loss_train_pssm += loss_pssm.cpu().detach()
            loss_train_entropy += loss_entropy.cpu().detach()

            loss_train += loss.cpu().detach()

            if scheduler is not None:
                scheduler.step()

            if (ite + 1) % report_iter == 0:
                if dl_test is not None:
                    t2 = time.time()
                    loss_v = eval_net(net, dl_test, loss_fnc, device=device, plot_results=viz, use_loss_coord=use_loss_coord)
                    t3 = time.time()
                    if scheduler is None:
                        lr = optimizer.param_groups[0]['lr']
                    else:
                        lr = scheduler.get_last_lr()[0]
                    LOG.info(
                        '{:6d}/{:6d}  Loss(training): {:6.4f}  Loss(test): {:6.4f}  Loss(pssm): {:6.4f}  Loss(entropy): {:6.4f}  LR: {:.8}  Time(train): {:.2f}s  Time(test): {:.2f}s  Time(total): {:.2f}h  ETA: {:.2f}h'.format(
                            ite + 1,int(max_iter), loss_train/report_iter, loss_v, loss_train_pssm/report_iter, loss_train_entropy/report_iter, lr, t2-t1, t3 - t2, (t3 - t0)/3600,(max_iter-ite+1)/(ite+1)*(t3-t0)/3600))
                    t1 = time.time()
                    loss_train_pssm = 0
                    loss_train_entropy = 0
                    loss_train = 0
            if (ite + 1) % checkpoint == 0:
                filename = "{:}checkpoint.pt".format(save)
                save_checkpoint(ite + 1, net.state_dict(), optimizer.state_dict(), filename=filename)
                LOG.info("Checkpoint saved: {}".format(filename))
            ite += 1

            if ite >= max_iter:
                stop_run = True
                break
        if stop_run:
            break
    # plotfullprotein(cnn_pred, caa_pred, cbb_pred, cnn_target, caa_target, cbb_target)
    # plotcoordinates(pred, target_coord)
    return net


def eval_net(net, dl, loss_fnc, device='cpu', plot_results=False, save_results=False, use_loss_coord=True, weight=0):
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
        for i,(features, pssm, mask, entropy) in enumerate(dl):
            features = features.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True) # Note that this is the padding mask, and not the mask for targets that are not available.
            pssm = pssm.to(device, non_blocking=True)
            entropy = entropy.to(device, non_blocking=True)

            pssm_pred, entropy_pred = net(features,mask)

            loss_pssm = torch.mean(torch.sum(torch.norm(pssm - pssm_pred,dim=1),dim=-1))

            loss_entropy = torch.mean(torch.sum(torch.norm(entropy - entropy_pred,dim=1),dim=-1))

            loss = loss_pssm + loss_entropy

            loss_v += loss

    net.train()
    return loss_v/len(dl)




def net_prediction(net, dl, device='cpu', plot_results=False, save_results=False):
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
        dist_err_mean = 0
        for i,(seq, dists,mask, coords) in enumerate(dl):
            seq = seq.to(device, non_blocking=True)
            dists = move_tuple_to(dists, device, non_blocking=True)
            coords = move_tuple_to(coords, device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)  # Note that this is the padding mask, and not the mask for targets that are not available.
            dists_pred, coords_pred = net(seq,mask)
            _, coords_pred_tr, coords_tr = loss_tr_tuples(coords_pred, coords, return_coords=True)
            M = dists[0] != 0
            dist_err = torch.sum(torch.sqrt(torch.mean(((dists_pred[0] - dists[0]) * M) ** 2, dim=(1, 2))) * 10)
            dist_err_mean += dist_err
            if save_results:
                compare_distogram(dists_pred, dists, mask, save_results="{:}dist_{:}".format(save_results,i), error=dist_err)
                plotfullprotein(coords_pred_tr, coords_tr, save_results="{:}coord_{:}".format(save_results,i), error=dist_err)
        if plot_results :
            compare_distogram(dists_pred, dists, mask, plot_results=plot_results)
            plotfullprotein(coords_pred_tr, coords_tr, plot_results=plot_results)
        dist_err_mean /= len(dl.dataset)
        print("Average distogram error in angstrom = {:2.2f}".format(dist_err_mean))
    net.train()
    return

