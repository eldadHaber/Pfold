import os

import torch
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from src.IO import load_checkpoint
from src.dataloader import PadCollate
from src.dataloader_npz import Dataset_npz
from src.network import select_network
from src.optimization import net_prediction

import pandas as pd
desired_width = 1200
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=desired_width)


if __name__ == '__main__':
    network_path = 'D:/Pytorch/Pfold/pretrained_networks/network_11_26_17_32_57.pt'
    dataset = './data/casp11_test_TBM/'
    dataset_out = './results/figures/'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 1

    i_entropy = True
    feature_dim = 1
    i_seq = True
    i_pssm = True
    i_cov = False
    i_cov_all = False
    i_contact = False
    inpainting = False

    network = 'vnet'
    feature_dim = 1
    network_args = {
        'nblocks': 4,
        'nlayers_pr_block': 5,
        'channels': 280,
        'chan_out': 3,
        'stencil_size': 3,
        }
    network_args['chan_in'] = 41

    dataset_test = Dataset_npz(dataset, feature_dim=feature_dim, seq_flip_prop=0, i_seq=i_seq, i_pssm=i_pssm,
                               i_entropy=i_entropy, i_cov=i_cov, i_cov_all=i_cov_all, i_contact=i_contact,
                               inpainting=inpainting)

    pad_modulo = 8

    dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=PadCollate(pad_modulo=pad_modulo),
                                           drop_last=False)


    # Load network and set it in evaluate mode
    net = torch.load(network_path)
    # net = select_network(network,network_args,feature_dim,cross_dist=False)
    optimizer = optim.Adam(list(net.parameters()), lr=1e-2)
    scheduler = OneCycleLR(optimizer, 1e-2, total_steps=2000000, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85,
                                        max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0)
    # net, optimizer, scheduler, ite_start = load_checkpoint(net, optimizer, scheduler, network_path)


    net.eval()
    net.to(device)

    # Check output folder is non-existent, and then create it
    os.makedirs(dataset_out,exist_ok=True)

    # net_prediction(net, dl_test, device=device, plot_results=False, save_results=False)
    net_prediction(net, dl_test, device=device, plot_results=False, save_results="{:}/".format(dataset_out))

