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


class Dataset_a3m(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        self.path = path
        assert os.path.isdir(path), "The path specified does not exist: {:}".format(path)
        a3m_files = []
        pdb_files = []
        #Find all a3m files in folder
        for file in os.listdir(path):
            if file.endswith(".a3m"):
                a3m_files.append(os.path.join(path, file))
                # Ensure that each a3m file have a corresponding pdb file
                filename, _ = os.path.splitext(file)
                pdb_file = os.path.join(path,filename + '.pdb')
                assert os.path.isfile(pdb_file), "the pdb file {:} does not exist".format(pdb_file)
                pdb_files.append(pdb_file)
        self.a3m_files = a3m_files
        self.pdb_files = pdb_files

        self.transform = transform
        self.target_transform = target_transform
        self.pdbparser = PDBParser(PERMISSIVE=False, QUIET=False)

    def __getitem__(self, index):
        msa = parse_a3m(self.a3m_files[index])
        target = self.pdbparser.get_structure("test", self.pdb_files[index])
        # target = parse_pdb(self.pdb_files[index])

        if self.transform is not None:
            msa = self.transform(msa)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return msa, target

    def __len__(self):
        return len(self.a3m_files)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.path + ')'




def parse_a3m(filename):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    for line in open(filename, "r"):
        # skip labels
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))
    #TODO FIX THIS SO IT USES THE DICT IN UTILS!!!!!!!!!!
    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    return msa
