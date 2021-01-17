import time

import matplotlib

from supervised.IO import save_checkpoint
from supervised.loss import loss_tr_tuples, Loss_reg_min_separation, LossMultiTargets
from supervised.utils import move_tuple_to
from supervised.visualization import compare_distogram, plotfullprotein
from supervised.config import config as c, load_from_config
import logging
logger = logging.getLogger('runner')

# from torch_lr_finder import LRFinder

matplotlib.use('Agg')

import torch

def train(net, optimizer, dataloader_train, loss_fnc, LOG=logger, device=None, dl_test=None, ite=0, max_iter=None, report_iter=None, checkpoint=None, scheduler=None, exp_dist_loss=None, result_dir=None, use_loss_coord=None, viz=None, loss_reg_fnc=None):
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

    device = load_from_config(device,'device')
    max_iter = load_from_config(max_iter,'max_iter')
    report_iter = load_from_config(report_iter,'report_iter')
    checkpoint = load_from_config(checkpoint,'checkpoint')
    exp_dist_loss = load_from_config(exp_dist_loss,'exp_dist_loss')
    result_dir = load_from_config(result_dir,'result_dir')
    use_loss_coord = load_from_config(use_loss_coord,'use_loss_coord')
    viz = load_from_config(viz,'viz')

    stop_run = False
    net.to(device)
    t0 = time.time()
    t1 = time.time()
    loss_train_d = 0
    loss_train_c = 0
    loss_train_reg = 0
    loss_train = 0
    loss_reg_min_sep_fnc = Loss_reg_min_separation()
    # loss_reg_min_sep_fnc = LossMultiTargets(inner_loss_reg_min_sep_fnc)
    best_v_loss = 9e9
    while True:
        for i, vars in enumerate(dataloader_train):
            features = vars[0][0]
            dists = vars[1]
            coords = vars[2]
            mask = vars[-1]
            features = features.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True) # Note that this is the padding mask, and not the mask for targets that are not available.
            dists = move_tuple_to(dists, device, non_blocking=True)
            coords = move_tuple_to(coords, device, non_blocking=True)

            w = ite / max_iter

            optimizer.zero_grad()
            dists_pred, coords_pred = net(features,mask)

            loss_d = loss_fnc(dists_pred, dists)
            if coords_pred is not None and exp_dist_loss<0 and use_loss_coord:
                loss_c = loss_tr_tuples(coords_pred, coords)
                loss_train_c += loss_c.cpu().detach()
                loss = (1-w)/2 * loss_d + (w+1)/2 * loss_c
            else:
                loss = loss_d
            if coords_pred is not None and loss_reg_fnc:
                seq = torch.argmax(features[:,0:20,:],dim=1)
                loss_reg = 10 * loss_reg_fnc(seq, coords_pred, mask)
                loss_train_reg += loss_reg.cpu().detach()
                loss += loss_reg
            if coords_pred is not None and loss_reg_min_sep_fnc:
                loss_reg_min_sep = 100 * loss_reg_min_sep_fnc(dists_pred,mask)
                loss += loss_reg_min_sep
                loss_train_reg += loss_reg_min_sep.cpu().detach()

            loss.backward()
            optimizer.step()
            loss_train_d += loss_d.cpu().detach()
            loss_train += loss.cpu().detach()

            if scheduler is not None:
                scheduler.step()

            if (ite + 1) % report_iter == 0:
                if dl_test is not None:
                    t2 = time.time()
                    loss_v, dist_err_ang, dist_err_ang_alq = eval_net(net, dl_test, loss_fnc, device=device, plot_results=viz, use_loss_coord=use_loss_coord, weight=w)
                    t3 = time.time()
                    if scheduler is None:
                        lr = optimizer.param_groups[0]['lr']
                    else:
                        lr = scheduler.get_last_lr()[0]
                    LOG.info(
                        '{:6d}/{:6d}  Loss(training): {:6.4f}%  Loss(test): {:6.4f}%  Loss(dist): {:6.4f}%  Loss(coord): {:6.4f}%  Loss(reg): {:6.4f}  Dist_err(ang): {:2.6f}  Dist_err(alq): {:2.6f}  LR: {:.8}  Time(train): {:.2f}s  Time(test): {:.2f}s  Time(total): {:.2f}h  ETA: {:.2f}h'.format(
                            ite + 1,int(max_iter), loss_train/report_iter*100, loss_v*100, loss_train_d/report_iter*100, loss_train_c/report_iter*100, loss_train_reg/report_iter, dist_err_ang, dist_err_ang_alq, lr, t2-t1, t3 - t2, (t3 - t0)/3600,(max_iter-ite+1)/(ite+1)*(t3-t0)/3600))
                    t1 = time.time()
                    loss_train_d = 0
                    loss_train_c = 0
                    loss_train_reg = 0
                    loss_train = 0
                    if loss_v < best_v_loss:
                        filename = "{:}/best_model_state.pt".format(result_dir)
                        save_checkpoint(ite + 1, net.state_dict(), optimizer.state_dict(), filename=filename)
                        torch.save(net, "{:}/best_model.pt".format(result_dir))
                        best_v_loss = loss_v
            if (ite + 1) % checkpoint == 0:
                filename = "{:}/checkpoint.pt".format(result_dir)
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
        dist_err_mean = 0
        dist_err_mean_alq = 0
        for i, vars in enumerate(dl):
            features = vars[0][0]
            dists = vars[1]
            coords = vars[2]
            mask = vars[-1]
            features = features.to(device, non_blocking=True)
            dists = move_tuple_to(dists, device, non_blocking=True)
            coords = move_tuple_to(coords, device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)  # Note that this is the padding mask, and not the mask for targets that are not available.
            dists_pred, coords_pred = net(features,mask)
            nb = features.shape[0]

            loss_d = loss_fnc(dists_pred, dists)
            if coords_pred is not None and use_loss_coord:
                loss_c, coords_pred_tr, coords_tr = loss_tr_tuples(coords_pred, coords, return_coords=True)
                loss = (1 - weight) / 2 * loss_d + (weight + 1) / 2 * loss_c
            else:
                loss = loss_d
            loss_v += loss * nb
            M = dists[0] != 0
            L = torch.sum(mask,dim=1)
            dist_err = torch.sum(torch.sqrt(torch.sum(((dists_pred[0] - dists[0]) * M) ** 2, dim=(1, 2)))/(L*L) * 10)
            dist_err_mean += dist_err

            dist_err_alq = torch.sum(torch.sqrt(torch.sum(((dists_pred[0] - dists[0]) * M) ** 2, dim=(1, 2)))/(L*(L-1)) * 10)
            dist_err_mean_alq += dist_err_alq

            if save_results:
                compare_distogram(dists_pred, dists, mask, save_results="{:}dist_{:}".format(save_results,i))
                plotfullprotein(coords_pred_tr, coords_tr, save_results="{:}coord_{:}".format(save_results,i))
        if plot_results :
            compare_distogram(dists_pred, dists, mask, plot_results=plot_results)
            plotfullprotein(coords_pred_tr, coords_tr, plot_results=plot_results)
    net.train()
    return loss_v/len(dl.dataset), dist_err_mean/len(dl.dataset), dist_err_mean_alq/len(dl.dataset)




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
        dist_err_mean = 0
        dist_err_RMSD_mean = 0
        dist_err_RMSD2_mean = 0
        dist_err_mean_alq = 0
        for i,(seq, dists,mask, coords,ids) in enumerate(dl):
            # if i!=23:
            #     continue
            seq = seq.to(device, non_blocking=True)
            dists = move_tuple_to(dists, device, non_blocking=True)
            coords = move_tuple_to(coords, device, non_blocking=True)
            mask_padding_and_unknown = (coords[0][:,0,:] != 0)
            mask = mask.to(device, non_blocking=True)  # Note that this is the padding mask, and not the mask for targets that are not available.
            dists_pred, coords_pred = net(seq,mask)
            _, coords_pred_tr, coords_tr = loss_tr_tuples(coords_pred, coords, return_coords=True)

            DpA = dists_pred[0]*10 #Dist predicted in Angstrom
            DtA = dists[0]*10 #Dist true in Angstrom
            M = dists[0] != 0
            L = torch.sum(mask_padding_and_unknown,dim=1)
            L2 = torch.sum(mask,dim=1)
            dist_err = torch.sum(torch.sum(torch.abs(DpA - DtA) * M, dim=(1,2))/(L*(L-1)))
            dist_err_mean += dist_err

            dist_err_alq = torch.sum(torch.sqrt(torch.sum(((DpA - DtA) * M) ** 2, dim=(1, 2)))/(L*(L-1)))
            dist_err_mean_alq += dist_err_alq

            dist_err_RMSD = torch.sum(torch.sqrt(torch.sum((DpA - DtA)**2 * M, dim=(1,2))/(L*L)))
            dist_err_RMSD_mean += dist_err_RMSD

            dist_err_RMSD2 = torch.sum(torch.sqrt(torch.sum((DpA - DtA)**2 * M, dim=(1,2))/(L2*L2)))
            dist_err_RMSD2_mean += dist_err_RMSD2

            if save_results:
                compare_distogram(dists_pred, dists, mask, save_results="{:}dist_{:}_ID_{:}".format(save_results,i,str(ids[0])), error=dist_err)
                plotfullprotein(coords_pred_tr, coords_tr, save_results="{:}coord_{:}_ID_{:}".format(save_results,i,str(ids[0])), error=dist_err)
        if plot_results :
            # compare_distogram(dists_pred, dists, mask, plot_results=plot_results)
            plotfullprotein(coords_pred_tr, coords_tr, plot_results=plot_results)
        dist_err_mean /= len(dl.dataset)
        dist_err_mean_alq /= len(dl.dataset)
        dist_err_RMSD_mean /= len(dl.dataset)
        dist_err_RMSD2_mean /= len(dl.dataset)

        print("Average distogram error in angstrom = {:2.2f}".format(dist_err_mean))
        print("Average distogram error in angstrom according to Alq = {:2.5f}".format(dist_err_mean_alq))
        print("Average distogram error in angstrom according to Jin = {:2.5f}".format(dist_err_RMSD_mean))
        print("Average distogram error in angstrom according to Jin = {:2.5f}, when L is the full protein length irregardless of unknown parts".format(dist_err_RMSD2_mean))
    net.train()
    return

