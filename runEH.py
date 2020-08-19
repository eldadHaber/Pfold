import os
import matplotlib.pyplot as plt
import string
import numpy as np
from numpy.linalg import norm
import copy
import random
import re
import os
import torch
import torchvision.transforms as transforms
import os.path as osp
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
#from Bio.PDB import PDBParser
from src.dataloader_utils import AA_DICT, MASK_DICT, DSSP_DICT, NUM_DIMENSIONS, AA_PAD_VALUE, MASK_PAD_VALUE, \
    DSSP_PAD_VALUE, SeqFlip, PSSM_PAD_VALUE, ENTROPY_PAD_VALUE, COORDS_PAD_VALUE, ListToNumpy


#def proj_3d(v1,v2):
#    #project v1 onto v2
#    return np.dot(v1,v2)/norm(v2) * v2


def ang2plain(v1,v2,v3,v4):
    nA = np.cross(v1,v2)
    nA = nA/np.linalg.norm(nA)
    nB = np.cross(v3,v4)
    nB = nB/np.linalg.norm(nB)

    cosPsi = np.dot(nA,nB)

    return np.degrees(np.arccos(cosPsi))



def convert_coord_to_dist_angles_linear(r1,r2,r3,mask=None):
    '''
    Data should be coordinate data in pnet format, meaning that each amino acid is characterized by a 3x3 matrix, which are the coordinates of r1,r2,r3=N,Calpha,Cbeta.
    This lite version, only computes the angles in along the sequence (upper one-off diagonal of the angle matrices of the full version)
    :return:
    '''
    seq_len = len(r1)

    d = np.zeros([seq_len, seq_len])
    phi = np.zeros([seq_len,seq_len])
    omega = np.zeros([seq_len,seq_len])
    theta = np.zeros([seq_len,seq_len])
    for i in range(seq_len):
        for j in range(i+1,seq_len):
            if mask is not None and (mask[i] == 0 or mask[j] == 0):
                continue

            r1i = r1[i]
            r2i = r2[i]
            r3i = r3[i]
            r1j = r1[j]
            r2j = r2[j]
            r3j = r3[j]

            # Compute distance Cb-Cb
            v1 = r3j - r3i
            d[i, j] = norm(v1)
            d[j, i] = d[i,j]

            # Compute phi
            v1 = r2i - r3i # Ca1 - Cb1
            v2 = r3i - r3j # Cb1 - Cb2
            phi[i,j] = np.degrees(np.arccos(np.dot(v1,v2)/norm(v1)/norm(v2)))

            v1 = r2j - r3j # Ca2 - Cb2
            v2 = r3j - r3i # Cb2 -Cb1
            phi[j, i] = np.degrees(np.arccos(np.dot(v1, v2) / norm(v1) / norm(v2)))

            # Thetas
            v1 = r1i - r2i  # N1 - Ca1
            v2 = r2i - r3i  # Ca1 - Cb1
            v3 = r3i - r3j  # Cb1 - Cb2
            theta[i,j] = ang2plain(v1, v2, v2, v3)

            v1 = r1j - r2j  # N2 - Ca2
            v2 = r2j - r3j  # Ca2 - Cb2
            v3 = r3j - r3i  # Cb2 - Cb1
            theta[j,i] = ang2plain(v1, v2, v2, v3)

            # Omega
            v1 = r2i - r3i # Ca1 - Cb1
            v2 = r3i - r3j # Cb1 - Cb2
            v3 = r3j - r2j # Cb2 - Ca2
            omega[i,j] = ang2plain(v1,v2,v2,v3)
            omega[j,i] = omega[i,j]

    return d,omega,phi,theta

def separate_coords(full_coords, pos):  # pos can be either 0(n_term), 1(calpha), 2(cterm)
    res = []
    for i in range(len(full_coords[0])):
        if i % 3 == pos:
            res.append([full_coords[j][i] for j in range(3)])

    return res

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

def flip_multidimensional_list(list_in):  # pos can be either 0(n_term), 1(calpha), 2(cterm)
    list_out = []
    ld = len(list_in)
    for i in range(len(list_in[0])):
        list_out.append([list_in[j][i] for j in range(ld)])
    return list_out


class ListToNumpy(object):
    def __init__(self):
        pass
    def __call__(self, *args):
        args_array = ()
        for arg in args:
            args_array += (np.asarray(arg),)
        return args_array

    def __repr__(self):
        return self.__class__.__name__ + '()'


def read_record(file_, num_evo_entries):
    """ Read all protein records from pnet file. """

    id = []
    seq = []
    pssm = []
    entropy = []
    dssp = []
    coord = []
    mask = []


    while True:
        next_line = file_.readline()
        for case in switch(next_line):
            if case('[ID]' + '\n'):
                id.append(file_.readline()[:-1])
            elif case('[PRIMARY]' + '\n'):
                seq.append(letter_to_num(file_.readline()[:-1], AA_DICT))
            elif case('[EVOLUTIONARY]' + '\n'):
                evolutionary = []
                for residue in range(num_evo_entries):
                    evolutionary.append([float(step) for step in file_.readline().split()])
                pssm.append(evolutionary)
                entropy.append([float(step) for step in file_.readline().split()])
            elif case('[SECONDARY]' + '\n'):
                dssp.append(letter_to_num(file_.readline()[:-1], DSSP_DICT))
            elif case('[TERTIARY]' + '\n'):
                tertiary = []
                for axis in range(NUM_DIMENSIONS): tertiary.append([float(coord) for coord in file_.readline().split()])
                coord.append(tertiary)
            elif case('[MASK]' + '\n'):
                mask.append(letter_to_num(file_.readline()[:-1], MASK_DICT))
            elif case(''):
                return id,seq,pssm,entropy,dssp,coord,mask

def parse_pnet(file):
    with open(file, 'r') as f:
        id, seq, pssm, entropy, dssp, coords, mask = read_record(f, 20)
        r1 = []
        r2 = []
        r3 = []
        pssm2 = []
        for i in range(len(pssm)): #We transform each of these, since they are inconveniently stored
            pssm2.append(flip_multidimensional_list(pssm[i]))
            r1.append(separate_coords(coords[i], 0))
            r2.append(separate_coords(coords[i], 1))
            r3.append(separate_coords(coords[i], 2))

    return id, seq, pssm2, entropy, dssp, r1,r2,r3, mask


##
dataFile ='./Data/testing'

id, seq, pssm2, entropy, dssp, r1, r2, r3, mask = parse_pnet(dataFile)

print('Done Reading File')

L2np = ListToNumpy()

S, RN, RCa, RCb, msk = L2np(seq[10],r1[10],r2[10],r3[10],mask[10])

d,omega,phi,theta = convert_coord_to_dist_angles_linear(RN,RCa,RCb,mask=msk)

plt.figure(1)
plt.imshow(d)
plt.figure(2)
plt.plot(omega)
plt.plot(phi)
plt.plot(theta)



print('Done Converting')

