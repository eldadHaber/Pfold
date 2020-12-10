import os
from datetime import datetime

import matplotlib
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from src import log
from src.IO import load_checkpoint
from src.dataloader import select_dataset
from src.loss import MSELoss, LossMultiTargets, EMSELoss, Loss_reg
from src.network import select_network
from src.optimization import eval_net, train
from src.utils import determine_network_param
from src.utils import fix_seed

matplotlib.use('Agg') #TkAgg

import pandas as pd
desired_width = 600
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=desired_width)

def main(c):
    #Initialize things
    c.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    fix_seed(c.seed) #Set a seed, so we make reproducible results.
    c.result_dir = "{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
        root='results',
        runner_name=c.basefolder,
        date=datetime.now(),
    )

    os.makedirs(c.result_dir)
    logfile_loc = "{}/{}.log".format(c.result_dir, 'output')
    c.LOG = log.setup_custom_logger('runner',logfile_loc,c.mode)
    c.LOG.info('---------Listing all parameters-------')
    state = {k: v for k, v in c._get_kwargs()}
    for key, value in state.items():
        c.LOG.info("{:30s} : {}".format(key, value))

    # Load Dataset
    dl_train, dl_test = select_dataset(c.dataset_train,c.dataset_test,c.feature_dim,batch_size=c.batch_size, network=c.network, i_seq=c.i_seq, i_pssm=c.i_pssm, i_entropy=c.i_entropy, i_cov=c.i_cov, i_cov_all=c.i_cov_all, i_contact=c.i_contact,inpainting=c.inpainting, seq_flip_prop=c.seq_flip_prop, random_crop=c.random_crop, cross_dist=c.use_cross_dist, chan_out=c.network_args['chan_out'])
    c.network_args['chan_in'] = dl_train.dataset.chan_in
    c.LOG.info('Datasets loaded, train  has {} samples. Test has {} samples'.format(len(dl_train.dataset),len(dl_test.dataset)))

    # Select loss function for training
    if c.sigma < 0:
        loss_inner_fnc = MSELoss()
    else:
        loss_inner_fnc = EMSELoss(sigma=c.sigma)
    loss_fnc = LossMultiTargets(loss_inner_fnc)

    c.LOG.info('Date:{}'.format(datetime.now()))

    net = select_network(c.network,c.network_args,c.feature_dim,cross_dist=c.use_cross_dist)

    c.LOG.info('Initializing Net, which has {} trainable parameters.'.format(determine_network_param(net)))
    net.to(c.device)
    optimizer = optim.Adam(list(net.parameters()), lr=c.SL_lr)
    scheduler = OneCycleLR(optimizer, c.SL_lr, total_steps=c.max_iter, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85,
                                        max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0)
    if c.load_from_previous != "":
        net, optimizer, scheduler, ite_start = load_checkpoint(net, optimizer, scheduler, c.load_from_previous, lr=c.SL_lr)
        c.LOG.info("Loading Checkpoint {:}, starting from iteration {:}".format(c.load_from_previous,ite_start))
    else:
        ite_start = 0
    if c.use_loss_reg:
        data = np.load(c.load_binding_dists)
        d_mean = data['d_mean']
        d_std = data['d_std']
        loss_reg = Loss_reg(d_mean,d_std,device=c.device)
    else:
        loss_reg = None
    net = train(net, optimizer, dl_train, loss_fnc, c.LOG, device=c.device, dl_test=dl_test, max_iter=c.max_iter, report_iter=c.report_iter, scheduler=scheduler, sigma=c.sigma, checkpoint=c.checkpoint, save="{:}/".format(c.result_dir), use_loss_coord=c.use_loss_coord, viz=c.viz, ite=ite_start, loss_reg_fnc=loss_reg)
    torch.save(net, "{:}/network.pt".format(c.result_dir))
    eval_net(net, dl_test, loss_fnc, device=c.device, save_results="{:}/".format(c.result_dir))
    # torch.save(net.state_dict(), "{:}/network.pt".format(c.result_dir))
    print("Done")