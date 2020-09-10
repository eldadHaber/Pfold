import os
import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset

from srcOld.dataloader_a3m import Dataset_a3m
from srcOld.dataloader_lmdb import Dataset_lmdb
from srcOld.dataloader_pnet import Dataset_pnet
from srcOld.dataloader_utils import SeqFlip, ListToNumpy, ConvertPnetFeaturesTo2D, ConvertCoordToDists, \
    ConvertDistAnglesToBins, ConvertPnetFeaturesTo1D


def select_dataset(path_train,path_test,seq_len=300,type='2D',batch_size=1, network=None):
    '''
    This is a wrapper routine for various dataloaders.
    Currently supports:
        folders with a3m files/pdb.
        pnet files.
    #TODO implement a3m files without pdb for testing
    :param path: path of input folder/file
    :return:
    '''

    Flip = SeqFlip()
    if os.path.isdir(path_train):
        dataset_train = Dataset_a3m(path_train)
    elif os.path.isfile(path_train) and os.path.splitext(path_train)[1].lower() == '.pnet':
        if type == '2D':
            transform_train = transforms.Compose([Flip, ListToNumpy(), ConvertPnetFeaturesTo2D()])
            transform_target_train = transforms.Compose([Flip, ListToNumpy(), ConvertCoordToDists()])
            transform_mask_train = transforms.Compose([Flip, ListToNumpy()])
        elif type == '1D':
            transform_train = transforms.Compose([Flip, ListToNumpy(), ConvertPnetFeaturesTo1D()])
            transform_target_train = transforms.Compose([Flip, ListToNumpy(), ConvertCoordToDists()])
            transform_mask_train = transforms.Compose([Flip, ListToNumpy()])


        dataset_train = Dataset_pnet(path_train, transform=transform_train,transform_target=transform_target_train,transform_mask=transform_mask_train)
    elif os.path.isfile(path_train) and os.path.splitext(path_train)[1].lower() == '.lmdb':
        if type == '2D':
            transform_train = transforms.Compose([Flip, ConvertPnetFeaturesTo2D()])
        elif type == '1D':
            transform_train = transforms.Compose([Flip, ConvertPnetFeaturesTo1D()])
        transform_target_train = transforms.Compose([Flip])
        transform_mask_train = transforms.Compose([Flip])

        dataset_train = Dataset_lmdb(path_train, transform=transform_train, target_transform=transform_target_train, mask_transform=transform_mask_train)
    else:
        raise NotImplementedError("dataset not implemented yet.")
    if os.path.isdir(path_test):
        dataset_test = Dataset_a3m(path_test)
    elif os.path.isfile(path_test) and os.path.splitext(path_test)[1].lower() == '.pnet':
        if type == '2D':
            transform_test = transforms.Compose([ListToNumpy(), ConvertPnetFeaturesTo2D()])
            transform_target_test = transforms.Compose([ListToNumpy(), ConvertCoordToDists()])
            transform_mask_test = transforms.Compose([ListToNumpy()])
        elif type == '1D':
            transform_test = transforms.Compose([Flip, ListToNumpy(), ConvertPnetFeaturesTo1D()])
            transform_target_test = transforms.Compose([Flip, ListToNumpy(), ConvertCoordToDists()])
            transform_mask_test = transforms.Compose([Flip, ListToNumpy()])

        dataset_test = Dataset_pnet(path_test, transform=transform_test,transform_target=transform_target_test, transform_mask=transform_mask_test)
    elif os.path.isfile(path_test) and os.path.splitext(path_test)[1].lower() == '.lmdb':
        if type == '2D':
            transform_test = transforms.Compose([ConvertPnetFeaturesTo2D()])
        elif type == '1D':
            transform_test = transforms.Compose([ConvertPnetFeaturesTo1D()])
        dataset_test = Dataset_lmdb(path_test, transform=transform_test)
    else:
        raise NotImplementedError("dataset not implemented yet.")

    assert len(dataset_train) >= batch_size

    if network.lower() == 'vnet':
        pad_modulo = 8
    else:
        pad_modulo = 1


    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=PadCollate(pad_modulo=pad_modulo),
                                           drop_last=True)
    dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=min(batch_size,len(dataset_test)), shuffle=False, num_workers=0, collate_fn=PadCollate(pad_modulo=pad_modulo),
                                           drop_last=False)

    return dl_train, dl_test



def pad_numpy_array_to_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.shape[dim]
    return torch.cat([torch.from_numpy(vec), torch.zeros(*pad_size)], dim=dim)


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

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        return:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        nb = len(batch)
        nf = batch[0][0].shape[0]
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        max_len = int(self.pad_mod * np.ceil(max_len / self.pad_mod))


        features = torch.empty((nb,nf,max_len),dtype=torch.float32)
        targets = torch.empty((nb,max_len,max_len),dtype=torch.float32)
        coords = torch.empty((nb,max_len,3),dtype=torch.float32)
        masks = torch.ones((nb,max_len),dtype=torch.int64)
        for i,batchi in enumerate(batch):
            feature = batchi[0]
            pad_size = list(feature.shape)
            pad_size[self.dim] = max_len - pad_size[self.dim]
            features[i,:,:] = torch.cat([torch.from_numpy(feature), torch.zeros(*pad_size)],dim=self.dim)

            target = batchi[1][0]
            pad_size = list(target.shape)
            pad_size[0] = max_len - pad_size[0]
            targets[i,:,:] = torch.cat([torch.cat([torch.from_numpy(target), torch.zeros(*pad_size)],dim=0),torch.zeros((max_len,pad_size[0]))],dim=1)

            coord = torch.tensor(batchi[3])
            pad_size = list(coord.shape)
            pad_size[0] = max_len - pad_size[0]
            coords[i,:,:] = torch.cat([coord, torch.zeros(*pad_size)], dim=0)

            masks[i,feature.shape[self.dim]:] = 0
        coords = coords.transpose(1,2)
        return features, (targets,), masks, coords

    def __call__(self, batch):
        return self.pad_collate(batch)
