import os

import numpy as np
import torch
from torch.utils.data import Dataset
from supervised.dataloader_utils import AA_LIST
from supervised.dataloader_npz import Dataset_npz

def select_dataset(path_train,path_test,feature_dim,batch_size, pad_modulo=1, i_seq=True,i_pssm=True,i_entropy=True,i_cov=True,i_contact=True,o_rCa=True,o_rCb=True,o_rN=True, o_dist=True,flip_protein=0.5,AA_list=AA_LIST,log_units=-9):
    """
    This is a wrapper routine for selecting the right dataloader.
    Currently it only supports datasets saved in npz format.
    """
    assert feature_dim==1, "Feature dimensions different from 1 is not currently supported."

    if os.path.isdir(path_train):
        dataset_train = Dataset_npz(path_train, flip_protein=flip_protein, i_seq=i_seq, i_pssm=i_pssm, i_entropy=i_entropy, i_cov=i_cov, i_contact=i_contact, o_rCa=o_rCa,o_rCb=o_rCb, o_rN=o_rN, o_dist=o_dist, AA_list=AA_list, log_units=log_units)
    else:
        raise NotImplementedError("dataset not implemented yet.")

    # Test Dataset
    if os.path.isdir(path_test):
        dataset_test = Dataset_npz(path_test, flip_protein=0, i_seq=i_seq, i_pssm=i_pssm, i_entropy=i_entropy, i_cov=i_cov, i_contact=i_contact, o_rCa=o_rCa,o_rCb=o_rCb, o_rN=o_rN, o_dist=o_dist, AA_list=AA_list, log_units=log_units)
    else:
        raise NotImplementedError("dataset not implemented yet.")

    if o_dist:
        pad_var_list = list([[0, 1],[1,1],[0,1],[0]])
    else:
        pad_var_list = list([[0, 1],[0,1],[0]])


    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=min(batch_size,len(dataset_train)), shuffle=True, num_workers=0, collate_fn=PadCollate(pad_modulo=pad_modulo,pad_var_list=pad_var_list),
                                           drop_last=True)
    dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=min(batch_size,len(dataset_train)), shuffle=True, num_workers=0, collate_fn=PadCollate(pad_modulo=pad_modulo,pad_var_list=pad_var_list),
                                           drop_last=False)
    return dl_train, dl_test


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in a batch of sequences.
    pad_modulo details whether the sequences should be padded to specific modulo length, which is typically needed when dealing with networks that coarsen the input, like u-nets.
        pad_modulo=1 means that any length is acceptable, pad_modulo=8 means that the output length will be divisible by 8.
    mask_var and mask_dim specify which variable and dimension in said variable the padding mask should be created from.
    pad_var_list specifies the dimensions in each variable that should be padded. 1 == padding, 0 == no padding
    An example:
        pad_var_list = list([[0, 1],[1,1],[0,1],[0]]) has 4 variables the first 3 has two dimensions, while the last has 1 dimension and should not be padded
    """
    def __init__(self, pad_var_list, pad_modulo=1,mask_var=0,mask_dim=-1):
        self.pad_mod = pad_modulo
        self.mask_var = mask_var
        self.mask_dim = mask_dim
        self.pad_var_list = pad_var_list

    def __call__(self, batch):
        return self.pad_collate(batch)

    def pad_collate(self,batch):
        # ndim = batch[0][0].ndim
        # if ndim == 2:
        return self.pad_collate_1d(batch)
        # if ndim == 3:
        #     return self.pad_collate_2d(batch)

    def pad_collate_1d(self, data):
        """
        This functions collates our data and transforms it into torch format. numpy arrays are padded according to the longest sequence in the batch in all dimensions.
        The padding mask is created according to mask_var and mask_dim, and is appended as the last variable in the output.

        args:
            data - Tuple of length nb, where nb is the batch size.
            data[0] contains the first batch and will also be a tuple with length nv, equal to the number of variables in a batch.
            data[0][0] contains the first variable of the first batch, this should also be a tuple with length nsm equal to the number of samples in the variable, like R1,R2,R3 inside coords.
            data[0][0][0] contains the actual data, and should be a numpy array.
            If any numpy array of ndim=0 is encountered it is assumed to be a string object, in which case it is turned into a string rather than a torch object.
            The datatype of the output is inferred from the input.
        return:
            A tuple of variables containing the input variables in order followed by the mask.
            Each variable is itself a tuple of samples
        """
        # find longest sequence
        nb = len(data)    # Number of batches
        nv = len(data[0]) # Number of variables in each batch
        vars = ()
        masks = ()
        for i in range(nv):
            ndim = data[0][i][0].ndim
            max_shape = np.zeros(ndim,dtype=np.int)
            for j in range(ndim):
                max_shape[j] = max(map(lambda x: x[i][0].shape[j], data))
                if self.pad_var_list[i][j] == 1:
                    max_shape[j] = int(self.pad_mod * np.ceil(max_shape[j] / self.pad_mod))
            ns = len(data[0][i]) # Number of samples in each variable (Like R1,R2,R3 in coords)
            samples = ()
            for ii in range(ns):
                batches = ()
                for iii in range(nb):
                    if ndim == 0:
                        batches += (str(data[iii][i][ii]),)
                    else:
                        argi = torch.from_numpy(data[iii][i][ii].copy())
                        if i == self.mask_var:
                            mask = torch.ones(max_shape[self.mask_dim], dtype=torch.int64)
                            mask[argi.shape[self.mask_dim]:] = 0
                            masks += (mask[None,:],)
                        for j in range(ndim):
                            if self.pad_var_list[i][j] == 1:
                                pad_size = list(argi.shape)
                                pad_size[j] = max_shape[j] - pad_size[j]
                                argi = torch.cat([argi, torch.zeros(*pad_size)], dim=j)
                        batches += (argi[None,:],)
                if ndim == 0:
                    sample = batches
                else:
                    sample = torch.cat(batches,dim=0)
                samples += (sample,)
            vars += (samples,)
        masks = torch.cat(masks,dim=0)
        vars += (masks,)
        return vars



