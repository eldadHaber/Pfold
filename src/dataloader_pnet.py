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
from src.dataloader_utils import AA_DICT, MASK_DICT, DSSP_DICT, NUM_DIMENSIONS, AA_PAD_VALUE, MASK_PAD_VALUE, \
    DSSP_PAD_VALUE, SegResizeAndFlip, PSSM_PAD_VALUE, ENTROPY_PAD_VALUE, COORDS_PAD_VALUE


class Dataset_pnet(Dataset):
    def __init__(self, file, seq_len=200, transform=None, target_transform=None):
        id,seq,pssm,entropy,dssp,r1,r2,r3,mask = parse_pnet(file)
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

        self.resize = SegResizeAndFlip(seq_len)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        seq = self.seq[index]
        pssm = self.pssm[index]
        entropy = self.entropy[index]
        mask = self.mask[index]
        r1 = self.r1[index]
        r2 = self.r2[index]
        r3 = self.r3[index]

        if self.resize is not None:
            seq, crop_idx, flip = self.resize(seq,AA_PAD_VALUE)
            pssm, _, _ = self.resize(pssm,PSSM_PAD_VALUE,idx_start=crop_idx, flip=flip)
            entropy, _, _ = self.resize(entropy,ENTROPY_PAD_VALUE,idx_start=crop_idx, flip=flip)
            mask, _, _ = self.resize(mask,MASK_PAD_VALUE,idx_start=crop_idx, flip=flip)
            r1, _, _ = self.resize(r1,COORDS_PAD_VALUE,idx_start=crop_idx, flip=flip)
            r2, _, _ = self.resize(r2,COORDS_PAD_VALUE,idx_start=crop_idx, flip=flip)
            r3, _, _ = self.resize(r3,COORDS_PAD_VALUE,idx_start=crop_idx, flip=flip)

        seq_onehot = np.eye(len(AA_DICT))[seq]

        dist,omega,phi,theta = convert_coord_to_dist_angles(r1, r2, r3,mask=mask)
        target = (dist,omega,phi,theta)

        f1d = np.concatenate((seq_onehot,pssm,entropy[:,None]),axis=1)

        f2d = np.concatenate([np.tile(f1d[:, None, :], [1, f1d.shape[0], 1]),np.tile(f1d[None, :, :], [f1d.shape[0], 1, 1])], axis=-1)

        return f2d, target

    def __len__(self):
        return len(self.seq)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.file + ')'

def proj_3d(v1,v2):
    #project v1 onto v2
    return np.dot(v1,v2)/norm(v2) * v2

def convert_coord_to_dist_angles(r1,r2,r3,mask=None):
    '''
    Data should be coordinate data in pnet format, meaning that each amino acid is characterized by a 3x3 matrix, which are the coordinates of N,Calpha,Cbeta.
    :param coord:
    :return:
    '''
    seq_len = len(r1)

    d = np.zeros([seq_len, seq_len])
    phi = np.zeros([seq_len, seq_len])
    omega = np.zeros([seq_len, seq_len])
    theta = np.zeros([seq_len, seq_len])
    for i in range(seq_len):
        for j in range(seq_len):
            if mask[i] == 0 or mask[j] == 0 or i == j:
                continue

            r1i = r1[i]
            r2i = r2[i]
            r2j = r2[j]
            r3i = r3[i]
            r3j = r3[j]

            a = norm(r3i-r3j)
            b = norm(r3i-r2i)
            c = norm(r3j-r2i)

            phi[i,j] = np.degrees(np.arccos((a*a + b*b - c*c)/(2*a*b)))

            v1 = r3j - r3i
            v2 = r2i - r3i
            normal_vec = np.cross(v1,v2)
            v3 = r2j - r3j

            #Now we find thetas
            v4 = r1i - r2i
            v4_proj = proj_3d(v4, v2)
            v4_ort = v4 - v4_proj
            theta[i,j] = (np.degrees(np.arccos(np.dot(v4_ort,normal_vec) / (norm(v4_ort) * norm(normal_vec)))) + 90) % 360

            if i > j: #These two are symmetric so we only calculate half of them
                d[i,j] = norm(v1)
                #First project this vector on the vector running between Cbi Cbj
                v3_proj = proj_3d(v3,v1)
                v3_ort = v3 - v3_proj
                #Now find the angle between the normal vector and project vector and add 90 to make it to the plane
                omega[i,j] = (np.degrees(np.arccos(np.dot(v3_ort,normal_vec) / (norm(v3_ort) * norm(normal_vec)))) + 90) % 360


    d = d + d.transpose()
    omega = omega + omega.transpose()

    mask_nan = np.isnan(phi)
    phi[mask_nan] = 0

    mask_nan = np.isnan(theta)
    theta[mask_nan] = 0

    return d,omega,phi,theta


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


def convert_coord_to_dist_angles_mini(coords):
    coords_nterm = [separate_coords(full_coords, 0) for full_coords in coords]
    coords_calpha = [separate_coords(full_coords, 1) for full_coords in coords]
    coords_cterm = [separate_coords(full_coords, 2) for full_coords in coords]
    print("Length coords_calpha (# proteins): ", len(coords_cterm))
    print("Length coords_calpha[0] (# amino acids in protein[0]) : ", len(coords_cterm[0]))
    print("Length coords_calpha[0][0] (# coordinates cterm is represented by (x,y,z)): ", len(coords_cterm[0][0]))

    phis, psis = [], []  # phi always starts with a 0 and psi ends with a 0
    ph_angle_dists, ps_angle_dists = [], []
    for k in range(len(coords)):
        phi, psi = [0.0], []
        # Use our own functions inspired from bioPython
        for i in range(len(coords_calpha[k])):
            # Calculate phi, psi
            # CALCULATE PHI - Can't calculate for first residue
            if i > 0:
                phi.append(get_dihedral(coords_cterm[k][i - 1], coords_nterm[k][i], coords_calpha[k][i],
                                        coords_cterm[k][i]))  # my_calc

            # CALCULATE PSI - Can't calculate for last residue
            if i < len(coords_calpha[k]) - 1:
                psi.append(get_dihedral(coords_nterm[k][i], coords_calpha[k][i], coords_cterm[k][i],
                                        coords_nterm[k][i + 1]))  # my_calc

        # Add an extra 0 to psi (unable to claculate angle with next aa)
        psi.append(0)
        # Add protein info to register
        phis.append(phi)
        psis.append(psi)

    return phis,psis

def get_dihedral(coords1, coords2, coords3, coords4):
    """Returns the dihedral angle in degrees."""

    a1 = coords2 - coords1
    a2 = coords3 - coords2
    a3 = coords4 - coords3

    v1 = np.cross(a1, a2)
    if (v1 * v1).sum(-1) ** 0.5 != 0:
        v1 = v1 / (v1 * v1).sum(-1) ** 0.5
    v2 = np.cross(a2, a3)
    if (v2 * v2).sum(-1) ** 0.5 != 0:
        v2 = v2 / (v2 * v2).sum(-1) ** 0.5
    porm = np.sign((v1 * a3).sum(-1))
    if ((v1 ** 2).sum(-1) * (v2 ** 2).sum(-1)) ** 0.5 != 0:
        rad = np.arccos((v1 * v2).sum(-1) / ((v1 ** 2).sum(-1) * (v2 ** 2).sum(-1)) ** 0.5)
    else:
        rad = 0
    if not porm == 0:
        rad = rad * porm

    return np.degrees(rad)




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



#
# def convert_coord_to_dist_angles(coords):
#     '''
#     Data should be coordinate data in pnet format, meaning that each amino acid is characterized by a 3x3 matrix, which are the coordinates of N,Calpha,Cbeta.
#     :param coord:
#     :return:
#     '''
#     dists = []
#     phis = []
#     omegas = []
#     thetas = []
#     for coord in coords: #For each protein
#         seq_len = len(coord[0])//3
#
#         d = np.zeros([seq_len, seq_len])
#         phi = np.zeros([seq_len, seq_len])
#         omega = np.zeros([seq_len, seq_len])
#         theta = np.zeros([seq_len, seq_len])
#         for i in range(seq_len):
#             for j in range(seq_len):
#                 Cbi = np.array([coord[0][i*3+2], coord[1][i*3+2], coord[2][i*3+2]])
#                 Cbj = np.array([coord[0][j*3+2], coord[1][j*3+2], coord[2][j*3+2]])
#                 Cai = np.array([coord[0][i*3+1], coord[1][i*3+1], coord[2][i*3+1]])
#                 Ni = np.array([coord[0][i*3], coord[1][i*3], coord[2][i*3]])
#                 Nj = np.array([coord[0][j*3], coord[1][j*3], coord[2][j*3]])
#
#
#                 a = norm(Cbi-Cbj)
#                 b = norm(Cbi-Cai)
#                 c = norm(Cbj-Cai)
#
#                 phi[i,j] = np.degrees(np.arccos((a*a + b*b - c*c)/(2*a*b)))
#
#                 Caj = np.array([coord[0][j*3+1], coord[1][j*3+1], coord[2][j*3+1]])
#                 v1 = Cbj - Cbi
#                 v2 = Cai - Cbi
#                 normal_vec = np.cross(v1,v2)
#                 v3 = Caj - Cbj
#
#                 #Now we find thetas
#                 v4 = Ni - Cai
#                 v4_proj = proj_3d(v4, Cai - Cbi)
#                 v4_ort = v4 - v4_proj
#                 theta[i,j] = (np.degrees(np.arccos(np.dot(v4_ort,normal_vec) / (norm(v4_ort) * norm(normal_vec)))) + 90) % 360
#
#                 if i > j: #These two are symmetric so we only calculate half of them
#                     d[i,j] = norm(Cbi-Cbj)
#                     #First project this vector on the vector running between Cbi Cbj
#                     v3_proj = proj_3d(v3,v1)
#                     v3_ort = v3 - v3_proj
#                     #Now find the angle between the normal vector and project vector and add 90 to make it to the plane
#                     omega[i,j] = (np.degrees(np.arccos(np.dot(v3_ort,normal_vec) / (norm(v3_ort) * norm(normal_vec)))) + 90) % 360
#
#
#         d = d + d.transpose()
#         dists.append(d)
#         omega = omega + omega.transpose()
#         omegas.append(omega)
#
#         mask_nan = np.isnan(phi)
#         phi[mask_nan] = 0
#
#         mask_nan = np.isnan(theta)
#         theta[mask_nan] = 0
#
#         phis.append(phi)
#         thetas.append(theta)
#
#     return dists,omegas,phis,thetas
