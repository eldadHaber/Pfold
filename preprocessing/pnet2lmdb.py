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
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
from itertools import compress
import time

from src.dataloader_pnet import Dataset_pnet, ConvertCoordToDistAnglesVec
from src.dataloader_utils import ListToNumpy, AA_DICT, DSSP_DICT, NUM_DIMENSIONS, MASK_DICT


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
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

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(dataset, lmdb_name, write_frequency=5000, num_workers=0, db_size=1e10):
    data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)

    isdir = os.path.isdir(lmdb_name)

    print("Generate LMDB to %s" % lmdb_name)
    db = lmdb.open(lmdb_name, subdir=isdir,
                   map_size=int(db_size), readonly=False,
                   meminit=False, map_async=True)

    print(len(dataset), len(data_loader))
    txn = db.begin(write=True)
    for i,(data) in enumerate(data_loader):
        features, target, mask = data[0]
        # print(type(data), data)
        txn.put(u'{}'.format(i).encode('ascii'), dumps_pyarrow((features, target, mask)))
        if (i+1) % write_frequency == 0:
            print("[%d/%d]" % (i, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(i + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()



def separate_coords(full_coords, pos):  # pos can be either 0(n_term), 1(calpha), 2(cterm)
    res = []
    for i in range(len(full_coords[0])):
        if i % 3 == pos:
            res.append([full_coords[j][i] for j in range(3)])

    return res


def flip_multidimensional_list(list_in):  # pos can be either 0(n_term), 1(calpha), 2(cterm)
    list_out = []
    ld = len(list_in)
    for i in range(len(list_in[0])):
        list_out.append([list_in[j][i] for j in range(ld)])
    return list_out

class switch(object):
    """Switch statement for Python, based on recipe from Python Cookbook."""

    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:  # changed for v1.5
            self.fall = True
            return True
        else:
            return False


def letter_to_num(string, dict_):
    """ Convert string of letters to list of ints """
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')
    num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
    num = [int(i) for i in num_string.split()]
    return num

def letter_to_bool(string, dict_):
    """ Convert string of letters to list of bools """
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')
    num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
    num = [bool(int(i)) for i in num_string.split()]
    return num



def read_pnet_into_lmdb(pnet_file, lmdb_file,max_seq_len=300,num_evo_entries=20,db_size=1e10,write_freq=5000,report_freq=5000):
    """ Read all protein records from pnet file. """

    isdir = os.path.isdir(lmdb_file)

    print("Generate LMDB to %s" % lmdb_file)
    db = lmdb.open(lmdb_file, subdir=isdir,
                   map_size=int(db_size), readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)

    t0 = time.time()
    cnt = 0
    n_excluded = 0
    with open(pnet_file, 'r') as f:
        while True:
            next_line = f.readline()
            for case in switch(next_line):
                if case('[ID]' + '\n'):
                    id = f.readline()[:-1]
                elif case('[PRIMARY]' + '\n'):
                    seq = letter_to_num(f.readline()[:-1], AA_DICT)
                elif case('[EVOLUTIONARY]' + '\n'):
                    evolutionary = []
                    for residue in range(num_evo_entries):
                        evolutionary.append([float(step) for step in f.readline().split()])
                    pssm = evolutionary
                    entropy = [float(step) for step in f.readline().split()]
                elif case('[SECONDARY]' + '\n'):
                    dssp = letter_to_num(f.readline()[:-1], DSSP_DICT)
                elif case('[TERTIARY]' + '\n'):
                    tertiary = []
                    for axis in range(NUM_DIMENSIONS): tertiary.append([float(coord) for coord in f.readline().split()])
                    coord = tertiary
                elif case('[MASK]' + '\n'):
                    mask = letter_to_bool(f.readline()[:-1], MASK_DICT)
                    if max_seq_len < len(seq):
                        n_excluded += 1
                        continue

                    features, target, mask = process_data(seq, pssm, entropy, coord, mask)
                    txn.put(u'{}'.format(cnt).encode('ascii'), dumps_pyarrow((features, target, mask)))
                    if (cnt+1) % write_freq == 0:
                        print("[%d]" % (cnt+1))
                        txn.commit()
                        txn = db.begin(write=True)
                    if (cnt+1) % report_freq == 0:
                        print("loading sample: {:}, excluded: {:}, Time: {:2.2f}".format(cnt+1,n_excluded, time.time() - t0))
                    cnt += 1
                elif case(''):
                    # finish iterating through dataset
                    txn.commit()
                    keys = [u'{}'.format(k).encode('ascii') for k in range(cnt)]
                    with db.begin(write=True) as txn:
                        txn.put(b'__keys__', dumps_pyarrow(keys))
                        txn.put(b'__len__', dumps_pyarrow(len(keys)))

                    print("Flushing database ...")
                    db.sync()
                    db.close()

                    return

def process_data(seq,pssm,entropy,coord,mask):
    pssm = flip_multidimensional_list(pssm)
    r1 = separate_coords(coord, 0)
    r2 = separate_coords(coord, 1)
    r3 = separate_coords(coord, 2)
    ltn = ListToNumpy()
    convert = ConvertCoordToDistAnglesVec()
    seq, pssm, entropy, mask, r1, r2, r3 = ltn(seq, pssm, entropy, mask, r1, r2, r3)

    dist, omega, phi, theta = convert(r1, r2, r3, mask)

    target = (dist, omega, phi, theta)
    features = (seq, pssm, entropy)
    return features, target, mask







if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("-f", "--folder", type=str, default='./../data/training_100.pnet')
    # parser.add_argument('-s', '--split', type=str, default="val")
    # parser.add_argument('--out', type=str, default="e:/test_lmdb")
    parser.add_argument("-f", "--folder", type=str, default='./../data/testing.pnet')
    parser.add_argument('-s', '--split', type=str, default="val")
    parser.add_argument('--out', type=str, default="e:/small_test_lmdb")
    parser.add_argument('-p', '--procs', type=int, default=0)
    seq_len = 500
    args = parser.parse_args()
    read_pnet_into_lmdb(args.folder, args.out, max_seq_len=seq_len, db_size=1e9, report_freq=1000, write_freq=5000)