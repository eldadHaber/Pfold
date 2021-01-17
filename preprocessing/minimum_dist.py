import os
import glob

import numpy as np
import torch.utils.data as data

from supervised.dataloader import PadCollate
from supervised.dataloader_npz import Dataset_npz
from supervised.dataloader_utils import SeqFlip, DrawFromProbabilityMatrix, MaskRandomSubset, convert_seq_to_onehot, \
    convert_1d_features_to_2d, ConvertCoordToDists, Random2DCrop, AA_LIST
import matplotlib.pyplot as plt
import matplotlib

from supervised.network_transformer import tr2DistSmall

matplotlib.use('TkAgg') #TkAgg

import pandas as pd
desired_width = 600
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=desired_width)

import torch

if __name__ == '__main__':
    dataset = './../data/casp11_testing/'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    batch_size = 1
    n_acids = 20

    flip_protein = 0
    i_seq = True
    i_pssm = False
    i_entropy = False
    i_cov = False
    i_contact = False
    o_rCa = True
    o_rCb = False
    o_rN = False
    o_dist = False
    AA_list = AA_LIST
    log_units = -9

    dataset_test = Dataset_npz(dataset, flip_protein, i_seq, i_pssm, i_entropy, i_cov, i_contact, o_rCa,
             o_rCb, o_rN, o_dist, AA_list, log_units=log_units)
    pad_var_list = list([[0, 0], [0, 0], [0]])
    dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, drop_last=False,collate_fn=PadCollate(pad_modulo=1,pad_var_list=pad_var_list))

    Dmins = torch.zeros((10000000))
    counter = 0
    fig = plt.figure(num=1, figsize=[15, 10])
    n_problems = 0


    for i, vars in enumerate(dl_test):
        onehotseq = vars[0][0]
        seq = torch.argmax(onehotseq,dim=1)
        coords = vars[1][0]
        seq = torch.squeeze(seq)
        c = coords
        mask = np.squeeze(c[:,0,:] != 0)

        c_s = c[:,:,mask]
        seq_s = seq[mask]

        dist_fnc = ConvertCoordToDists()
        D = tr2DistSmall(c_s)
        idx = D > 0
        Dmin = torch.min(D[idx])
        Dmins[i] = Dmin

        if (i+1) % 100 == 0:
            print("{:} samples. Average minimum distance = {:2.4f}+-{:2.4f}, smallest one = {:2.4f}".format(i+1,torch.mean(Dmins[0:i]),torch.std(Dmins[0:i]),torch.min(Dmins[0:i])))

    print("{:} samples. Average minimum distance = {:2.4f}+-{:2.4f}, smallest one = {:2.4f}".format(i+1,torch.mean(Dmins[0:i]),torch.std(Dmins[0:i]),torch.min(Dmins[0:i])))
