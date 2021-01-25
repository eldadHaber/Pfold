import os
import glob
import torch
import numpy as np
import torch.utils.data as data

from supervised.dataloader import PadCollate
from supervised.dataloader_npz import Dataset_npz
from supervised.dataloader_utils import SeqFlip, DrawFromProbabilityMatrix, MaskRandomSubset, ConvertCoordToDists, \
    Random2DCrop, AA_LIST
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use('TkAgg') #TkAgg
desired_width = 600
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=desired_width)

def count_nn_dist(dataloader,max_count=1000000,report_iter=100000):
    """
    This function goes through a dataset and counts up all the neighbouring distances in the proteins, each nn-dist is saved according to the two amino acids in question.
    """

    D = np.zeros((max_count,n_acids,n_acids),dtype=np.float32)
    counters = np.zeros((n_acids,n_acids),dtype=np.int32)
    fig = plt.figure(num=1, figsize=[15, 10])
    n_problems = 0
    for i, vars in enumerate(dataloader):
        onehotseq = vars[0][0]
        seq = torch.argmax(onehotseq,dim=1)
        coords = vars[1][0]
        seq = torch.squeeze(seq)

        c = coords
        mask = np.squeeze(c[:,0,:] != 0)

        d = torch.squeeze(torch.norm(c[:, :, 1:] - c[:, :, :-1], 2, dim=1))

        for j in range(len(seq)-1):
            if mask[j] == 0 or mask[j+1] == 0:
                continue
            dist = d[j]
            # if dist < 0.3 or dist > 0.5:
            #     n_problems += 1
            #     continue
            idx1 = seq[j]
            idx2 = seq[j+1]
            if idx1 < idx2:
                counts = counters[idx1, idx2]
                D[counts,idx1,idx2] = dist
                counters[idx1, idx2] += 1
            else:
                counts = counters[idx2, idx1]
                D[counts,idx2,idx1] = dist
                counters[idx2, idx1] += 1
            if (np.sum(counters)+1) % report_iter == 0:
                print("{:}".format(np.sum(counters)+1))
                D_mean = np.sum(D,axis=0) / (counters+1e-10)
                D_std = np.empty_like(D_mean)
                for ii in range(D_std.shape[0]):
                    for jj in range(ii,D_std.shape[0]):
                        D_std[ii,jj] = np.std(D[:counters[ii,jj],ii,jj])
                m = counters > 0
                vmin = np.min(D_mean[m])
                vmax = np.max(D_mean[m])
                plt.clf()
                plt.subplot(1,3,1)
                plt.imshow(D_mean,vmin=vmin,vmax=vmax)
                plt.colorbar()
                plt.title("# problems {:}".format(n_problems))
                plt.subplot(1,3,2)
                varmin = np.min(D_std[m])
                varmax = np.max(D_std[m])
                plt.imshow(D_std, vmin=varmin, vmax=varmax)
                plt.colorbar()
                plt.title("std")
                plt.subplot(1,3,3)
                cmin = np.min(counters[m])
                cmax = np.max(counters[m])
                plt.imshow(counters,vmin=cmin,vmax=cmax)
                plt.colorbar()
                plt.title("{:}".format(np.sum(counters)+1))
                plt.pause(1)
                save = "{:}.png".format("dist")
                fig.savefig(save)

    print("{:}".format(np.sum(counters)+1))
    D_mean = np.sum(D,axis=0) / (counters+1e-10)
    D_std = np.empty_like(D_mean)
    for ii in range(D_std.shape[0]):
        for jj in range(ii,D_std.shape[0]):
            D_std[ii,jj] = np.std(D[:counters[ii,jj],ii,jj])
    m = counters > 0
    return D, D_mean, D_std, counters, m



if __name__ == '__main__':
    dataset = './../data/casp11_training_90/'
    # dataset = './../data/casp11_testing/'
    output_folder = './../data/'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    batch_size = 1
    n_acids = 20

    flip_protein = 0
    i_seq = True
    i_pssm = False
    i_entropy = False
    i_inpaint = False
    i_cov = False
    i_contact = False
    o_rCa = True
    o_rCb = False
    o_rN = False
    o_dist = False
    AA_list = AA_LIST
    log_units = -9

    dataset_test = Dataset_npz(dataset, flip_protein, i_seq, i_pssm, i_entropy, i_inpaint, i_cov, i_contact, o_rCa, o_rCb, o_rN, o_dist, AA_list, log_units=log_units)
    pad_var_list = list([[0, 0], [0, 0], [0]])
    dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, drop_last=False,collate_fn=PadCollate(pad_modulo=1,pad_var_list=pad_var_list))

    # Check output folder is non-existent, and then create it
    # os.makedirs(dataset_out,exist_ok=True)
    D, D_mean, D_std, counters, m = count_nn_dist(dl_test, max_count=1000000, report_iter=100000)

    np.savez('{:}nn-distances'.format(output_folder),distances_mean=D_mean,distances_std=D_std,counters=counters,log_units=log_units,AA_LIST=AA_list)

    fig = plt.figure(num=1, figsize=[15, 10])
    vmin = np.min(D_mean[m])
    vmax = np.max(D_mean[m])
    plt.clf()
    plt.subplot(1,3,1)
    plt.imshow(D_mean,vmin=vmin,vmax=vmax)
    plt.colorbar()
    # plt.title("# problems {:}".format(n_problems))
    plt.subplot(1,3,2)
    varmin = np.min(D_std[m])
    varmax = np.max(D_std[m])
    plt.imshow(D_std, vmin=varmin, vmax=varmax)
    plt.colorbar()
    plt.title("std")
    plt.subplot(1,3,3)
    cmin = np.min(counters[m])
    cmax = np.max(counters[m])
    plt.imshow(counters,vmin=cmin,vmax=cmax)
    plt.colorbar()
    plt.title("{:}".format(np.sum(counters) + 1))
    plt.pause(1)
    save = "{:}{:}.png".format(output_folder,"dist_done")
    fig.savefig(save)

