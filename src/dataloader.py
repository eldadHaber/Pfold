import string
import numpy as np
import copy
import random
import os
import torch
import torchvision.transforms as transforms
import os.path as osp
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from Bio.PDB import PDBParser

from src.dataloader_a3m import Dataset_a3m
from src.dataloader_pnet import Dataset_pnet


def select_dataset(path):
    '''
    This is a wrapper routine for various dataloaders.
    Currently supports:
        folders with a3m files/pdb.
        pnet files.
    #TODO implement a3m files without pdb for testing
    :param path: path of input folder/file
    :return:
    '''
    if os.path.isdir(path):
        dataset = Dataset_a3m(path)
    elif os.path.isfile(path) and os.path.splitext(path)[1].lower() == '.pnet':
        dataset = Dataset_pnet(path)
    else:
        raise NotImplementedError("dataset not implemented yet.")

    dl_train = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                                           drop_last=True)

    return dl_train




# def parse_pdb(filename):
#     for line in