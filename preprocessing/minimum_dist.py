import os
import glob

import numpy as np
import torch.utils.data as data

from src.dataloader_utils import SeqFlip, DrawFromProbabilityMatrix, MaskRandomSubset, convert_seq_to_onehot, \
    convert_1d_features_to_2d, ConvertCoordToDists, Random2DCrop
import matplotlib.pyplot as plt
import matplotlib

from src.network_transformer import tr2DistSmall

matplotlib.use('TkAgg') #TkAgg

import pandas as pd
desired_width = 600
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=desired_width)

import torch

if __name__ == '__main__':
    dataset = './../data/casp11_training_90/'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    batch_size = 1
    n_acids = 20

    i_entropy = False
    feature_dim = 1
    i_seq = True
    i_pssm = False
    i_cov = False
    i_cov_all = False
    i_contact = False
    inpainting = False


    class Dataset_npz(data.Dataset):
        '''
        Reads a folder full of npz files, and treats its as a database.
        Expects the data to be packed as follows:
        (features, target, mask)
            features = (seq, pssm, entropy)
            target = (r1, r2, r3)

        '''

        def __init__(self, folder, seq_flip_prop=0.5, chan_out=3, feature_dim=1, i_seq=False, i_pssm=False,
                     i_entropy=False, i_cov=False, i_cov_all=False, i_contact=False, inpainting=False,
                     random_crop=False, cross_dist=False):

            search_command = folder + "*.npz"
            npzfiles = [f for f in glob.glob(search_command)]
            self.folder = folder
            self.files = npzfiles

            self.Flip = SeqFlip(seq_flip_prop)
            self.chan_out = chan_out
            self.draw_seq_from_pssm = False
            self.draw = DrawFromProbabilityMatrix(fraction_of_seq_drawn=0.2)
            self.draw_prob = 0.5
            self.seq_mask = MaskRandomSubset()
            self.feature_dim = feature_dim
            self.i_seq = i_seq
            self.i_pssm = i_pssm
            self.i_entropy = i_entropy
            self.i_cov = i_cov
            self.i_cov_all = i_cov_all
            self.i_contact = i_contact
            self.inpainting = inpainting
            self.chan_in = self.calculate_chan_in()
            self.coord_to_dist = ConvertCoordToDists()
            self.crop = Random2DCrop()
            self.use_crop = random_crop
            self.use_cross_dist = cross_dist

        def calculate_chan_in(self):
            assert (
                        self.i_cov is False or self.i_cov_all is False), "You can only have one of (i_cov, i_cov_all) = True"
            chan_in = 0
            if self.i_seq:
                chan_in += 20
            if self.i_pssm:
                chan_in += 20
            if self.i_entropy:
                chan_in += 1
            if self.inpainting:
                chan_in += 4  # 3 coordinates + 1 mask
            if self.feature_dim == 1:
                if self.i_cov_all:
                    chan_in += 10 * 441
                elif self.i_cov:
                    chan_in += 10 * 21
                if self.i_contact:
                    chan_in += 20
            elif self.feature_dim == 2:
                chan_in *= 2
                if self.i_cov_all:
                    chan_in += 441
                elif self.i_cov:
                    chan_in += 21
                if self.i_contact:
                    chan_in += 1
            print("Number of features used this run {:}".format(chan_in))
            return chan_in

        def __getitem__(self, index):
            data = np.load(self.files[index])
            if self.chan_out == 3:
                coords = (data['r1'],)
            elif self.chan_out == 6:
                coords = (data['r1'], data['r2'],)
            elif self.chan_out == 9:
                coords = (data['r1'], data['r2'], data['r3'],)
            else:
                raise NotImplementedError("The number of channels out is not supported.")
            seq = data['seq']

            return seq, coords

        def __len__(self):
            return len(self.files)

        def match_target_channels(self, target):
            if self.chan_out == 3:
                target = (target[0],)
            elif self.chan_out == 6:
                target = target[0:2]
            elif self.chan_out == 9:
                pass
            else:
                raise NotImplementedError("chan_out is {}, which is not implemented".format(self.chan_out))
            return target

        def __repr__(self):
            return self.__class__.__name__ + ' (' + self.folder + ')'



    dataset_test = Dataset_npz(dataset, feature_dim=feature_dim, seq_flip_prop=0, i_seq=i_seq, i_pssm=i_pssm,
                               i_entropy=i_entropy, i_cov=i_cov, i_cov_all=i_cov_all, i_contact=i_contact,
                               inpainting=inpainting)

    dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    # Check output folder is non-existent, and then create it
    # os.makedirs(dataset_out,exist_ok=True)

    Dmins = torch.zeros((10000000))
    counter = 0
    fig = plt.figure(num=1, figsize=[15, 10])
    n_problems = 0


    for i, (seq, coords) in enumerate(dl_test):
        seq = torch.squeeze(seq)

        c = coords[0]
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




        # ii = torch.argmin(D[idx])
        # idx_flat = torch.flatten(idx)
        # cumidx = torch.cumsum(idx_flat,dim=0)
        # lin_idx = torch.where(cumidx == ii)[0].numpy()
        # np.unravel_index(lin_idx,idx.shape)




