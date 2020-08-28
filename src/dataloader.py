import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from src.dataloader_a3m import Dataset_a3m
from src.dataloader_lmdb import Dataset_lmdb
from src.dataloader_pnet import Dataset_pnet
from src.dataloader_utils import SeqFlip, ConvertPnetFeaturesTo2D, ConvertDistAnglesToBins, ListToNumpy, \
    ConvertCoordToDistAnglesVec


def select_dataset(path_train,path_test,seq_len=300):
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
        dataset_train = Dataset_a3m(path_train)
    elif os.path.isfile(path_train) and os.path.splitext(path_train)[1].lower() == '.pnet':

        transform_train = transforms.Compose([SeqFlip(), ListToNumpy(), ConvertPnetFeaturesTo2D()])
        transform_target_train = transforms.Compose([SeqFlip(), ListToNumpy(), ConvertCoordToDistAnglesVec(), ConvertDistAnglesToBins()])
        transform_mask_train = transforms.Compose([SeqFlip()])

        dataset_train = Dataset_pnet(path_train, transform=transform_train,transform_target=transform_target_train,transform_mask=transform_mask_train)
    elif os.path.isfile(path_train) and os.path.splitext(path_train)[1].lower() == '.lmdb':
        transform_train = transforms.Compose([SeqFlip(), ConvertPnetFeaturesTo2D()])
        transform_target_train = transforms.Compose([SeqFlip(), ConvertDistAnglesToBins()])
        transform_mask_train = transforms.Compose([SeqFlip()])
        dataset_train = Dataset_lmdb(path_train, transform=transform_train, target_transform=transform_target_train, mask_transform=transform_mask_train)
    else:
        raise NotImplementedError("dataset not implemented yet.")
    if os.path.isdir(path_test):
        dataset_test = Dataset_a3m(path_test)
    elif os.path.isfile(path_test) and os.path.splitext(path_test)[1].lower() == '.pnet':
        transform_test = transforms.Compose([ListToNumpy(), ConvertPnetFeaturesTo2D()])
        transform_target_test = transforms.Compose([ListToNumpy(), ConvertCoordToDistAnglesVec(), ConvertDistAnglesToBins()])

        dataset_test = Dataset_pnet(path_test, transform=transform_test,transform_target=transform_target_test)
    elif os.path.isfile(path_test) and os.path.splitext(path_test)[1].lower() == '.lmdb':
        transform_test = transforms.Compose([ConvertPnetFeaturesTo2D()])
        transform_target_test = transforms.Compose([ConvertDistAnglesToBins()])
        dataset_test = Dataset_lmdb(path_test, transform=transform_test, target_transform=transform_target_test)
    else:
        raise NotImplementedError("dataset not implemented yet.")

    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0,
                                           drop_last=True)
    dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0,
                                           drop_last=False)

    return dl_train, dl_test
