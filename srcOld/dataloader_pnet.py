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
from Bio.PDB import PDBParser
import time
from src.dataloader_utils import AA_DICT, MASK_DICT, DSSP_DICT, NUM_DIMENSIONS, AA_PAD_VALUE, MASK_PAD_VALUE, \
    DSSP_PAD_VALUE, SeqFlip, PSSM_PAD_VALUE, ENTROPY_PAD_VALUE, COORDS_PAD_VALUE, ListToNumpy
from src.visualization import compare_distogram
from itertools import compress

class Dataset_pnet(Dataset):
    def __init__(self, file, transform=None, transform_target=None, transform_mask=None, max_seq_len=300):
        id,seq,pssm,entropy,dssp,r1,r2,r3,mask = parse_pnet(file,max_seq_len=max_seq_len)
        self.file = file
        self.id = id
        self.seq = seq
        self.pssm = pssm
        self.entropy = entropy
        self.dssp = dssp
        self.mask = mask
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3

        self.transform = transform
        self.transform_target = transform_target
        self.transform_mask = transform_mask
        self.nfeatures = 84

    def __getitem__(self, index):
        features = (self.seq[index], self.pssm[index], self.entropy[index])
        mask = self.mask[index]
        target = (self.r1[index], self.r2[index], self.r3[index], mask)

        if self.transform is not None:
            features = self.transform(features)
        if self.transform_target is not None:
            target = self.transform_target(target)
        if self.transform_mask is not None:
            mask = self.transform_mask(mask) #TODO CHECK THAT THIS IS NOT DOUBLE FLIPPED!

        return features, target, mask

    def __len__(self):
        return len(self.seq)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.file + ')'


def convert_dist_angles_to_bins(d,omega,phi,theta):
    d_bin = np.linspace(250,2000,36)
    mask_unknown = d == 0
    d_cat = np.digitize(d, bins=d_bin)

    angle25_bin = np.linspace(15,360,24)
    angle13_bin = np.linspace(15,180,12)

    d_mask = d_cat == 36
    omega_cat = np.digitize(omega, bins=angle25_bin)
    omega_cat[d_mask] = 24

    theta_cat = np.digitize(theta, bins=angle25_bin)
    theta_cat[d_mask] = 24

    phi_cat = np.digitize(phi, bins=angle13_bin)
    phi_cat[d_mask] = 12

    #Now we make sure that all the unknown gets set to -100, which is the standard in pytorch for ignored values
    d_cat[mask_unknown] = -100
    omega_cat[mask_unknown] = -100
    theta_cat[mask_unknown] = -100
    phi_cat[mask_unknown] = -100


    # compare_distogram((d,omega,theta,phi), (d_cat,omega_cat,theta_cat,phi_cat))

    return d_cat,omega_cat,phi_cat,theta_cat



def ang_between_planes(v1,v2,v3,v4):
    '''
    Given a plane spanned by: (v1,v2) and another plane spanned by (v3,v4), it finds the angle between these two planes.
    The angle is found by computing the normal vector to each plane, and finding the angle between those two vectors
    :param v1:
    :param v2:
    :param v3:
    :param v4:
    :return:
    '''
    nA = np.cross(v1,v2)
    nA = nA/np.linalg.norm(nA)
    nB = np.cross(v3,v4)
    nB = nB/np.linalg.norm(nB)

    cosPsi = np.round(np.dot(nA,nB),decimals=6)
    if cosPsi < -1 or cosPsi > 1:
        print('wtf')

    return np.degrees(np.arccos(cosPsi))



def crossProdMat(V1,V2):
    Vcp = np.zeros(V1.shape)
    Vcp[:,:,0] =  V1[:,:,1]*V2[:,:,2] - V1[:,:,2]*V2[:,:,1]
    Vcp[:,:,1] = -V1[:,:,0]*V2[:,:,2] + V1[:,:,2]*V2[:,:,0];
    Vcp[:,:,2] =  V1[:,:,0]*V2[:,:,1] - V1[:,:,1]*V2[:,:,0];
    return Vcp

def ang_between_planes_matrix(v1,v2,v3,v4):
    nA = crossProdMat(v1,v2)
    nB = crossProdMat(v3, v4)
    nA = nA/(np.sqrt(np.sum(nA**2,axis=2))[:,:,None])
    nB = nB/(np.sqrt(np.sum(nB**2,axis=2))[:,:,None])

    cosPsi = np.sum(nA*nB,axis=2)
    #Psi    = torch.acos(cosPsi)
    return cosPsi


def ang_between_planes_matrix_360(v1,v2,v3,v4):
    nA = crossProdMat(v1,v2)
    nB = crossProdMat(v3, v4)
    nA = nA/(np.sqrt(np.sum(nA**2,axis=2))[:,:,None])
    nB = nB/(np.sqrt(np.sum(nB**2,axis=2))[:,:,None])

    v2n = v2/(np.sqrt(np.sum(v2**2,axis=2))[:,:,None])
    det = np.sum(v2n*crossProdMat(nA,nB), axis=2)
    dot = np.sum(nA*nB,axis=2)
    angle = np.degrees(np.arctan2(det,dot))+180

    #Psi    = torch.acos(cosPsi)
    return angle



class ConvertCoordToDistAnglesVec(object):
    def __init__(self):
        pass
    def __call__(self, *args):
        if len(args) == 1:
            args = args[0]

        rN = args[0]
        rCa = args[1]
        rCb = args[2]
        mask = args[3]

        # Get D
        D = np.sum(rCb ** 2, axis=1)[:,None] + np.sum(rCb ** 2, axis=1)[None,:] - 2 * (rCb @ rCb.transpose())
        M = mask[:,None] @  mask[None,:]
        D = np.sqrt(np.maximum(M*D,0))

        # Get Upper Phi
        # TODO clean Phi to be the same as OMEGA
        V1x = (rCa[:, 0])[:,None] - (rCb[:, 0])[:,None]
        V1y = (rCa[:, 1])[:,None] - (rCb[:, 1])[:,None]
        V1z = (rCa[:, 2])[:,None] - (rCb[:, 2])[:,None]
        V2x = (rCb[:, 0])[:,None] - (rCb[:, 0])[:,None].transpose()
        V2y = (rCb[:, 1])[:,None] - (rCb[:, 1])[:,None].transpose()
        V2z = (rCb[:, 2])[:,None] - (rCb[:, 2])[:,None].transpose()
        # Normalize them
        V1n = np.sqrt(V1x**2 + V1y**2 + V1z**2)
        V1x = V1x/V1n
        V1y = V1y/V1n
        V1z = V1z/V1n
        V2n = np.sqrt(V2x**2 + V2y**2 + V2z**2)
        V2x = V2x/V2n
        V2y = V2y/V2n
        V2z = V2z/V2n
        # go for it
        PHI = M*(V1x * V2x + V1y * V2y + V1z * V2z)
        PHI = np.degrees(np.arccos(PHI))
        indnan = np.isnan(PHI)
        PHI[indnan] = 0.0

        # Omega
        nat = rCa.shape[0]
        V1 = np.zeros((nat, nat, 3))
        V2 = np.zeros((nat, nat, 3))
        V3 = np.zeros((nat, nat, 3))
        # Ca1 - Cb1
        V1[:,:,0] = ((rCa[:,0])[:,None] - (rCb[:,0])[:,None]).repeat(nat,axis=1)
        V1[:,:,1] = ((rCa[:,1])[:,None] - (rCb[:,1])[:,None]).repeat(nat,axis=1)
        V1[:,:,2] = ((rCa[:,2])[:,None] - (rCb[:,2])[:,None]).repeat(nat,axis=1)
        # Cb1 - Cb2
        V2[:,:,0] = (rCb[:,0])[:,None] - (rCb[:,0])[:,None].transpose()
        V2[:,:,1] = (rCb[:,1])[:,None] - (rCb[:,1])[:,None].transpose()
        V2[:,:,2] = (rCb[:,2])[:,None] - (rCb[:,2])[:,None].transpose()
        # Cb2 - Ca2
        V3[:,:,0] = ((rCb[:,0])[None,:] - (rCa[:,0])[None,:]).repeat(nat,axis=0)
        V3[:,:,1] = ((rCb[:,1])[None,:] - (rCa[:,1])[None,:]).repeat(nat,axis=0)
        V3[:,:,2] = ((rCb[:,2])[None,:] - (rCa[:,2])[None,:]).repeat(nat,axis=0)

        OMEGA     = M*ang_between_planes_matrix_360(V1, V2, V2, V3)
        indnan = np.isnan(OMEGA)
        OMEGA[indnan] = 0.0

        # Theta
        V1 = np.zeros((nat, nat, 3))
        V2 = np.zeros((nat, nat, 3))
        V3 = np.zeros((nat, nat, 3))
        # N - Ca
        V1[:,:,0] = (rN[:,0][:,None] - rCa[:,0][:,None]).repeat(nat,axis=1)
        V1[:,:,1] = (rN[:,1][:,None] - rCa[:,1][:,None]).repeat(nat,axis=1)
        V1[:,:,2] = (rN[:,2][:,None] - rCa[:,2][:,None]).repeat(nat,axis=1)
        # Ca - Cb # TODO - repeated computation
        V2[:,:,0] = (rCa[:,0][:,None] - rCb[:,0][:,None]).repeat(nat,axis=1)
        V2[:,:,1] = (rCa[:,1][:,None] - rCb[:,1][:,None]).repeat(nat,axis=1)
        V2[:,:,2] = (rCa[:,2][:,None] - rCb[:,2][:,None]).repeat(nat,axis=1)
        # Cb1 - Cb2 # TODO - repeated computation
        V3[:,:,0] = rCb[:,0][:,None] - rCb[:,0][:,None].transpose()
        V3[:,:,1] = rCb[:,1][:,None] - rCb[:,1][:,None].transpose()
        V3[:,:,2] = rCb[:,2][:,None] - rCb[:,2][:,None].transpose()

        THETA = M*ang_between_planes_matrix_360(V1, V2, V2, V3)
        indnan = np.isnan(THETA)
        THETA[indnan] = 0.0
        return D, OMEGA, PHI, THETA

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



def read_record(file_, num_evo_entries):
    """ Read all protein records from pnet file. """

    id = []
    seq = []
    pssm = []
    entropy = []
    dssp = []
    coord = []
    mask = []

    t0 = time.time()
    while True:
        next_line = file_.readline()
        for case in switch(next_line):
            if case('[ID]' + '\n'):
                id.append(file_.readline()[:-1])
                if len(id) % 1000 == 0:
                    print("loading sample: {:}, Time: {:2.2f}".format(len(id),time.time() - t0))
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
                mask.append(letter_to_bool(file_.readline()[:-1], MASK_DICT))
            elif case(''):


                return id,seq,pssm,entropy,dssp,coord,mask

def parse_pnet(file,max_seq_len=-1):
    with open(file, 'r') as f:
        t0 = time.time()
        id, seq, pssm, entropy, dssp, coords, mask = read_record(f, 20)
        print("loading data complete! Took: {:2.2f}".format(time.time()-t0))
        r1 = []
        r2 = []
        r3 = []
        pssm2 = []
        for i in range(len(pssm)): #We transform each of these, since they are inconveniently stored
            pssm2.append(flip_multidimensional_list(pssm[i]))
            r1.append(separate_coords(coords[i], 0))
            r2.append(separate_coords(coords[i], 1))
            r3.append(separate_coords(coords[i], 2))
            if i+1 % 1000 == 0:
                print("flipping and separating: {:}, Time: {:2.2f}".format(len(id), time.time() - t0))

        args = (id, seq, pssm2, entropy, dssp, r1,r2,r3, mask)
        if max_seq_len > 0:
            filter = np.full(len(seq), True, dtype=bool)
            for i,seq_i in enumerate(seq):
                if len(seq_i) > max_seq_len:
                    filter[i] = False
            new_args = ()
            for list_i in (id, seq, pssm2, entropy, dssp, r1,r2,r3, mask):
                new_args += (list(compress(list_i,filter)),)
        else:
            new_args = args
        print("parse complete! Took: {:2.2f}".format(time.time() - t0))
    return new_args

