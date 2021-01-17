import os
from datetime import datetime
from supervised.config import config as c


import matplotlib
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from supervised import log
from supervised.IO import load_checkpoint
from supervised.dataloader import select_dataset
from supervised.log import log_all_parameters
from supervised.loss import MSELoss, LossMultiTargets, EMSELoss, Loss_reg, load_loss_reg
from supervised.network import select_network
from supervised.optimization import eval_net, train
from supervised.utils import determine_network_param
from supervised.utils import fix_seed
import pandas as pd
desired_width = 600
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=desired_width)



def main():
    c['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #Initialize things
    fix_seed(c['seed']) #Set a seed, so we make reproducible results.
    c['result_dir'] = "{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
        root='results',
        runner_name=c['basefolder'],
        date=datetime.now(),
    )

    os.makedirs(c['result_dir'])
    logfile_loc = "{}/{}.log".format(c['result_dir'], 'output')
    LOG = log.setup_custom_logger('runner',logfile_loc,c['mode'])

    # Load Dataset
    dl_train, dl_test = select_dataset(c['dataset_train'],c['dataset_test'],c['feature_dim'],batch_size=c['batch_size'], **c['data_args'])
    LOG.info('Datasets loaded, train  has {} samples. Test has {} samples'.format(len(dl_train.dataset),len(dl_test.dataset)))

    # Select loss function for training
    if c['exp_dist_loss'] < 0:
        loss_inner_fnc = MSELoss()
    else:
        loss_inner_fnc = EMSELoss(sigma=c['exp_dist_loss'])
    loss_fnc = LossMultiTargets(loss_inner_fnc)

    net = select_network(c['network'],c['feature_dim'],**c['network_args'])

    LOG.info('Initializing Net, which has {} trainable parameters.'.format(determine_network_param(net)))
    net.to(c['device'])
    optimizer = optim.Adam(list(net.parameters()), lr=c['SL_lr'])
    scheduler = OneCycleLR(optimizer, c['SL_lr'], total_steps=c['max_iter'], pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85,
                                        max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0)
    if c['load_from_previous'] != "":
        net, optimizer, scheduler, ite_start = load_checkpoint(net, optimizer, scheduler, c['load_from_previous'], lr=c['SL_lr'])
        LOG.info("Loading Checkpoint {:}, starting from iteration {:}".format(c['load_from_previous'],ite_start))
    else:
        ite_start = 0
    if c['use_loss_reg']:
        load_loss_reg(c['load_nn_dists'],AA_list=c['data_args']['AA_list'],log_units=c['data_args']['log_units'])
    else:
        loss_reg_fnc = None

    log_all_parameters(LOG, c)
    net = train(net, optimizer, dl_train, loss_fnc, dl_test=dl_test, scheduler=scheduler,ite=ite_start, loss_reg_fnc=loss_reg_fnc)
    torch.save(net, "{:}/network.pt".format(c.result_dir))
    eval_net(net, dl_test, loss_fnc, device=c.device, save_results="{:}/".format(c.result_dir))
    # torch.save(net.state_dict(), "{:}/network.pt".format(c.result_dir))
    print("Done")