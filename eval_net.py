import os

import torch
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from supervised.IO import load_checkpoint
from supervised.dataloader import PadCollate
from supervised.dataloader_npz import Dataset_npz
from supervised.network import select_network
from supervised.optimization import net_prediction

import pandas as pd
desired_width = 1200
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=desired_width)


if __name__ == '__main__':
    # network_path = 'D:/Pytorch/Pfold/pretrained_networks/2020-11-26_17_30_02/network.pt'
    # network_path = 'D:/Pytorch/run_amazon/2020-12-09_22_45_12/network.pt'
    # network_path = 'D:/Dropbox/ComputationalGenetics/results/data augmentation/no augmentation/2021-02-06_02_20_34/best_model_state.pt'
    # network_path = 'D:/Dropbox/ComputationalGenetics/results/data augmentation/no augmentation/2021-02-06_11_08_42/best_model_state.pt'
    # network_path = 'D:/Dropbox/ComputationalGenetics/results/data augmentation/no augmentation/2021-02-07_09_06_53/best_model_state.pt'
    # network_path = 'D:/Dropbox/ComputationalGenetics/results/data augmentation/no_weight/2021-02-05_15_39_25/best_model_state.pt'
    # network_path = '/home/tue/Dropbox/ComputationalGenetics/results/data augmentation/no_augmentation_including_subs/best_model_state.pt'
    # network_path = '/home/tue/Dropbox/ComputationalGenetics/results/data augmentation/no_augmentation_including_subs/best_model_state1.pt'
    # network_path = '/home/tue/Dropbox/ComputationalGenetics/results/data augmentation/New/Including Selected subproteins/2021-02-18_18_08_08/best_model_state.pt'
    # network_path = '/home/tue/Dropbox/ComputationalGenetics/results/data augmentation/New/Including Selected subproteins/2021-02-19_08_12_19/best_model_state.pt'
    # network_path = '/home/tue/Dropbox/ComputationalGenetics/results/data augmentation/New/Including Selected subproteins/2021-02-20_14_21_18/best_model_state.pt'
    network_path = '/home/tue/Dropbox/ComputationalGenetics/results/data augmentation/New/Parent Proteins/2021-02-21_16_01_18/best_model_state.pt'
    # dataset = './data/casp11_validation/'
    # dataset = './data/casp11_test_TBM/'
    dataset = './../../data/casp11_testing/'
    dataset_out = './results/figures/'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 1

    i_entropy = False
    feature_dim = 1
    i_seq = True
    i_pssm = False
    i_cov = False
    i_cov_all = False
    i_contact = False
    i_inpaint = False
    o_rCa = True
    o_rCb = False
    o_rN = False
    o_dist = True
    log_units = -10
    AA_list = 'ACDEFGHIKLMNPQRSTVWY-'
    # network = 'vnet'
    # feature_dim = 1
    # network_args = {
    #     'nblocks': 4,
    #     'nlayers_pr_block': 5,
    #     'channels': 280,
    #     'chan_out': 3,
    #     'stencil_size': 3,
    #     }
    # network_args['chan_in'] = 41

    dataset_test = Dataset_npz(dataset, flip_protein=0, i_seq=i_seq, i_pssm=i_pssm, i_entropy=i_entropy, i_cov=i_cov, i_contact=i_contact, o_rCa=o_rCa, o_rCb=o_rCb, o_rN=o_rN, o_dist=o_dist, i_inpaint=i_inpaint, AA_list=AA_list, log_units=log_units, use_weight=False)
    pad_modulo = 8
    if o_dist:
        pad_var_list = list([[0, 1],[1,1],[0,1],[0],[0]])
    else:
        pad_var_list = list([[0, 1],[0,1],[0],[0]])



    dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=PadCollate(pad_modulo=pad_modulo,pad_var_list=pad_var_list), drop_last=False)

    ite_start, net, opt, lr_scheduler = load_checkpoint(network_path,device=device)

    # Load network and set it in evaluate mode
    # net = torch.load(network_path)
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
    # net_prediction(net, dl_test, device=device, plot_results=False, save_results="{:}/".format(dataset_out))
    net_prediction(net, dl_test, device=device, plot_results=False, save_results=False)
    # net_prediction(net, dl_test, device=device, plot_results=True, save_results=False)

