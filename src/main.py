import os
from datetime import datetime
import numpy as np
import matplotlib
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from src.utils import determine_network_param
from src import log
from src.dataloader import select_dataset
from src.loss import MSELoss, LossMultiTargets, EMSELoss
from src.network import select_network
from src.network_vnet import vnet1D
from src.network_transformer import TransformerModel
from src.optimization import train, eval_net
from src.utils import fix_seed

matplotlib.use('TkAgg') #TkAgg

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
    dl_train, dl_test = select_dataset(c.dataset_train,c.dataset_test,c.feature_dim,batch_size=c.batch_size, network=c.network, i_seq=c.i_seq, i_pssm=c.i_pssm, i_entropy=c.i_entropy, i_cov=c.i_cov, i_cov_all=c.i_cov_all, i_contact=c.i_contact,inpainting=c.inpainting, seq_flip_prop=c.seq_flip_prop)
    c.network_args['chan_in'] = dl_train.dataset.chan_in
    c.LOG.info('Datasets loaded, train  has {} samples. Test has {} samples'.format(len(dl_train.dataset),len(dl_test.dataset)))

    # Select loss function for training
    loss_inner_fnc = MSELoss()
    loss_fnc = LossMultiTargets(loss_inner_fnc)

    c.LOG.info('Date:{}'.format(datetime.now()))

    net = select_network(c.network,c.network_args,c.feature_dim)

    c.LOG.info('Initializing Net, which has {} trainable parameters.'.format(determine_network_param(net)))
    net.to(c.device)
    optimizer = optim.Adam(list(net.parameters()), lr=c.SL_lr)
    scheduler = OneCycleLR(optimizer, c.SL_lr, total_steps=c.max_iter, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85,
                                        max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0)

    net = train(net, optimizer, dl_train, loss_fnc, c.LOG, device=c.device, dl_test=dl_test, max_iter=c.max_iter, report_iter=c.report_iter, scheduler=scheduler)
    eval_net(net, dl_test, loss_fnc, device=c.device, save_results="{:}/".format(c.result_dir))
    torch.save(net, "{:}/network.pt".format(c.result_dir))
    # torch.save(net.state_dict(), "{:}/network.pt".format(c.result_dir))
    print("Done")
    input()
