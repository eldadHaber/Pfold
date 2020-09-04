import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from srcOld.dataloader_a3m import Dataset_a3m
from srcOld.dataloader_lmdb import Dataset_lmdb
from srcOld.dataloader_pnet import Dataset_pnet
from srcOld.dataloader_utils import SeqFlip, ListToNumpy, ConvertPnetFeaturesTo2D, ConvertCoordToDists, \
    ConvertDistAnglesToBins, ConvertPnetFeaturesTo1D


def select_dataset(path_train,path_test,seq_len=300,type='2D'):
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

    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0,
                                           drop_last=True)
    dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0,
                                           drop_last=False)

    return dl_train, dl_test
