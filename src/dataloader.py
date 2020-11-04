import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataloader_npz import Dataset_npz


def select_dataset(path_train,path_test,feature_dim=1,batch_size=1, network=None, i_seq=False, i_pssm=False, i_entropy=False, i_cov=False, i_cov_all=False, i_contact=False,inpainting=False, seq_flip_prop=0.5):
    '''
    This is a wrapper routine for various dataloaders.
    Currently supports:
        folders with a3m files/pdb.
        pnet files.
    #TODO implement a3m files without pdb for testing
    :param path: path of input folder/file
    :return:
    '''


    if os.path.isdir(path_train):
        dataset_train = Dataset_npz(path_train, feature_dim=feature_dim, seq_flip_prop=seq_flip_prop, i_seq=i_seq, i_pssm=i_pssm, i_entropy=i_entropy, i_cov=i_cov, i_cov_all=i_cov_all, i_contact=i_contact,inpainting=inpainting)
    else:
        raise NotImplementedError("dataset not implemented yet.")

    # Test Dataset
    if os.path.isdir(path_test):
        dataset_test = Dataset_npz(path_test, feature_dim=feature_dim, seq_flip_prop=0, i_seq=i_seq, i_pssm=i_pssm, i_entropy=i_entropy, i_cov=i_cov, i_cov_all=i_cov_all, i_contact=i_contact,inpainting=inpainting)
    else:
        raise NotImplementedError("dataset not implemented yet.")

    if network.lower() == 'vnet':
        pad_modulo = 8
    else:
        pad_modulo = 1


    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=min(batch_size,len(dataset_train)), shuffle=True, num_workers=0, collate_fn=PadCollate(pad_modulo=pad_modulo),
                                           drop_last=True)
    dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=min(batch_size,len(dataset_train)), shuffle=False, num_workers=0, collate_fn=PadCollate(pad_modulo=pad_modulo),
                                           drop_last=False)
    # dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=min(batch_size,len(dataset_test)), shuffle=True, num_workers=0, collate_fn=PadCollate(pad_modulo=pad_modulo),
    #                                        drop_last=False)

    return dl_train, dl_test


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=1, pad_modulo=1):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.pad_mod = pad_modulo

    def pad_collate(self,batch):
        ndim = batch[0][0].ndim
        if ndim == 2:
            return self.pad_collate_1d(batch)
        if ndim == 3:
            return self.pad_collate_2d(batch)

    def pad_collate_1d(self, batch):
        """
        args:
            batch - list of (features, targets)
        return:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        nb = len(batch)
        nf = batch[0][0].shape[0]
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        max_len = int(self.pad_mod * np.ceil(max_len / self.pad_mod))

        distances = ()
        nt = len(batch[0][1]) # Number of distograms
        tt = torch.empty((nt,nb,max_len,max_len),dtype=torch.float32)

        coords = ()
        nc = len(batch[0][2]) # Number of coordinates
        cc = torch.empty((nc,nb,3,max_len),dtype=torch.float32)

        features = torch.empty((nb,nf,max_len),dtype=torch.float32)
        masks = torch.ones((nb, max_len), dtype=torch.int64)

        for i,batchi in enumerate(batch):
            feature = batchi[0]
            pad_size = list(feature.shape)
            pad_size[self.dim] = max_len - pad_size[self.dim]
            features[i,:,:] = torch.cat([torch.from_numpy(feature.copy()), torch.zeros(*pad_size)],dim=self.dim)

            for j in range(nt):
                distance = batchi[1][j]
                pad_size = list(distance.shape)
                pad_size[0] = max_len - pad_size[0]
                tt[j,i,:,:] = torch.cat([torch.cat([torch.from_numpy(distance), torch.zeros(*pad_size)],dim=0),torch.zeros((max_len,pad_size[0]))],dim=1)

            for j in range(nc):
                coord = torch.from_numpy(batchi[2][j].copy())
                pad_size = list(coord.shape)
                pad_size[1] = max_len - pad_size[1]
                cc[j,i,:,:] = torch.cat([coord, torch.zeros(*pad_size)], dim=1)

            masks[i,feature.shape[self.dim]:] = 0
        for i in range(nt):
            distances += (tt[i,:,:,:],)
        for i in range(nc):
            coords += (cc[i,:,:,:],)
        return features, distances, masks, coords



    def pad_collate_2d(self, batch):
        """
        args:
            batch - list of (features, targets)
        return:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        nb = len(batch)
        nf = batch[0][0].shape[0]
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        max_len = int(self.pad_mod * np.ceil(max_len / self.pad_mod))

        distances = ()
        nt = len(batch[0][1]) # Number of distograms
        tt = torch.empty((nt,nb,max_len,max_len),dtype=torch.float32)

        coords = ()
        nc = len(batch[0][2]) # Number of coordinates
        cc = torch.empty((nc,nb,3,max_len),dtype=torch.float32)

        features = torch.empty((nb,nf,max_len, max_len),dtype=torch.float32)
        masks = torch.ones((nb, max_len), dtype=torch.int64)

        for i,batchi in enumerate(batch):
            feature = batchi[0]
            features[i,:,:,:] = self.pad_2d_tensor(feature,max_len)

            for j in range(nt):
                distance = batchi[1][j]
                pad_size = list(distance.shape)
                pad_size[0] = max_len - pad_size[0]
                tt[j,i,:,:] = torch.cat([torch.cat([torch.from_numpy(distance), torch.zeros(*pad_size)],dim=0),torch.zeros((max_len,pad_size[0]))],dim=1)

            for j in range(nc):
                coord = torch.from_numpy(batchi[2][j].copy())
                pad_size = list(coord.shape)
                pad_size[1] = max_len - pad_size[1]
                cc[j,i,:,:] = torch.cat([coord, torch.zeros(*pad_size)], dim=1)

            masks[i,feature.shape[self.dim]:] = 0
        for i in range(nt):
            distances += (tt[i,:,:,:],)
        for i in range(nc):
            coords += (cc[i,:,:,:],)
        return features, distances, masks, coords

    def pad_2d_tensor(self,a,max_len):
        pad_size = list(a.shape)
        pad_size[1] = max_len - pad_size[1]
        tmp = torch.cat([torch.from_numpy(a.copy()), torch.zeros(*pad_size)], dim=1)
        pad_size = list(tmp.shape)
        pad_size[2] = max_len - pad_size[2]
        return torch.cat([tmp, torch.zeros(*pad_size)], dim=2)

    def __call__(self, batch):
        return self.pad_collate(batch)