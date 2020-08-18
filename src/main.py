import os
from datetime import datetime
import numpy as np
import matplotlib
import torch
import torch.optim as optim

from src import log
from src.dataloader import select_dataset
from src.network import ResNet
from src.optimization import train
from src.utils import fix_seed

matplotlib.use('TkAgg') #TkAgg

# import pandas as pd
# desired_width = 600
# pd.set_option('display.width', desired_width)
# pd.set_option('display.max_columns', 10)
# np.set_printoptions(linewidth=desired_width)

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
    dl_train = select_dataset(c.dataset_train)

    # Select loss function for training
    loss_fnc = torch.nn.CrossEntropyLoss()

    c.LOG.info('Date:{}'.format(datetime.now()))

    net = ResNet(c.nlayers)
    net.to(c.device)
    optimizer = optim.Adam(list(net.parameters()), lr=c.SL_lr)

    net = train(net, optimizer, dl_train, loss_fnc, c.LOG, device=c.device)
