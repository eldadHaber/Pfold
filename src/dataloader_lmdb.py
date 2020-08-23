import os
import os.path as osp
import os, sys
import os.path as osp
from PIL import Image
import six
import string

import lmdb
import pickle
import msgpack
import tqdm
import pyarrow as pa
import re
import numpy as np

import torch
import torch.utils.data as data


class Dataset_lmdb(data.Dataset):
    '''
    Reads an lmdb database.
    Expects the data to be packed as follows:
    (features, target, mask)
        features = (seq, pssm, entropy)
        target = (dist, omega, phi, theta)

    '''
    def __init__(self, db_path, transform=None, target_transform=None, mask_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform
        self.nfeatures = 84

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        features = unpacked[0]

        targets = unpacked[1]

        mask = unpacked[2]

        if self.transform is not None:
            features = self.transform(features)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return features, targets, mask

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
