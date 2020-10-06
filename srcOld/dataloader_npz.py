import glob
import os.path as osp
import random
import sys
import time

import lmdb
import pyarrow as pa
import torch.utils.data as data
import copy

import numpy as np
from srcOld.dataloader_utils import SeqFlip, DrawFromProbabilityMatrix, MaskRandomSubset


class Dataset_npz(data.Dataset):
    '''
    Reads a folder full of npz files, and treats its as a database.
    Expects the data to be packed as follows:
    (features, target, mask)
        features = (seq, pssm, entropy)
        target = (r1, r2, r3)

    '''
    def __init__(self, folder, chan_in, transform=None, target_transform=None, mask_transform=None, chan_out=3, draw_seq_from_pssm=False, preload=False,mask_random_seq=True):

        search_command = folder + "*.npz"
        npzfiles = [f for f in glob.glob(search_command)]
        self.folder = folder
        self.files = npzfiles

        self.transform = transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform
        self.chan_out = chan_out
        self.chan_in = chan_in
        self.draw_seq_from_pssm = draw_seq_from_pssm
        self.draw = DrawFromProbabilityMatrix(fraction_of_seq_drawn=0.2)
        self.draw_prob = 0.5
        self.seq_mask = MaskRandomSubset()
        self.apply_mask = mask_random_seq

    def __getitem__(self, index):
        t0 = time.time()
        data = np.load(self.files[index]) #Probably need to folder structure as well here
        features = (data['seq'], data['pssm'], data['entropy'])
        targets = (data['r1'],data['r2'],data['r3'])
        # mask = data['mask']

        features = self.select_features(features,targets,mask_random_seq=self.apply_mask)
        targets = self.match_target_channels(targets)

        if isinstance(self.transform.transforms[0], SeqFlip):
            self.transform.transforms[0].reroll()
        if self.transform is not None:
            features = self.transform(features)

        if self.target_transform is not None:
            distances, coords = self.target_transform(targets)

        return features, distances, coords

    def __len__(self):
        return len(self.files)

    def select_features(self,features,targets,mask_random_seq=True):
        seq,pssm,entropy = features
        p = random.random()
        if self.chan_in == 21:
            if self.draw_seq_from_pssm and p > self.draw_prob:
                features = (self.draw(pssm, seq=seq),)
            else:
                features = (seq,)
        elif self.chan_in == 22:
            if self.draw_seq_from_pssm and p > self.draw_prob:
                features = (self.draw(pssm, seq=seq), entropy)
            else:
                features = (seq, entropy)
        elif self.chan_in == 25:
            #We use this for inpainting, channels are 1hot + coords + mask
            if mask_random_seq:
                r, m = self.seq_mask(targets[0])
            else:
                r = targets[0]
                m = np.ones(r.shape[1])
            features = (seq, r.T, m)
        elif self.chan_in == 41:
            features = (seq, pssm)
        elif self.chan_in == 42:
            features = (seq, pssm, entropy)
        else:
            raise NotImplementedError("The selected number of channels in is not currently supported")


        return features


    def match_target_channels(self,target):
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
