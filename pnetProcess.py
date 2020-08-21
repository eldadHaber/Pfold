import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
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


# Routines to read the file
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

# Processing the data to maps

def ang2plain(v1,v2,v3,v4):
    nA = torch.cross(v1,v2)
    nA = nA/torch.norm(nA)
    nB = torch.cross(v3,v4)
    nB = nB/torch.norm(nB)

    cosPsi = torch.dot(nA,nB)
    #Psi    = torch.acos(cosPsi)
    return cosPsi


def convertCoordToDistAngles(rN, rCa, rCb, mask=None):
    '''
    Data should be coordinate data in pnet format, meaning that each amino acid is characterized
    by a 3x3 matrix, which are the coordinates of r1,r2,r3=N,Calpha,Cbeta.
    This lite version, only computes the angles in along the sequence
    (upper one-off diagonal of the angle matrices of the full version)
    '''
    seq_len = rN.shape[0]
    # Initialize distances and angles

    d     = torch.zeros([seq_len, seq_len])
    phi   = torch.zeros([seq_len,seq_len])
    omega = torch.zeros([seq_len,seq_len])
    theta = torch.zeros([seq_len,seq_len])

    for i in range(seq_len):
        for j in range(i+1,seq_len):
            if mask is not None and (mask[i] == 0 or mask[j] == 0):
                continue

            r1i = rN[i, :]  # N1  atom
            r2i = rCa[i, :]  # Ca1 atom
            r3i = rCb[i, :]  # Cb1 atom
            r1j = rN[j, :]  # N2  atom
            r2j = rCa[j, :]  # Ca2 atom
            r3j = rCb[j, :]  # Cb2 atom

            # Compute distance Cb-Cb
            vbb = r3j - r3i
            d[i, j] = torch.norm(vbb)
            d[j, i] = d[i,j]

            # Compute phi
            v1 = r2i - r3i # Ca1 - Cb1
            v2 = r3i - r3j # Cb1 - Cb2
            #phi[i,j] = torch.acos(torch.dot(v1,v2)/torch.norm(v1)/torch.norm(v2))
            phi[i, j] = torch.dot(v1, v2) / torch.norm(v1) / torch.norm(v2)

            v1 = r2j - r3j # Ca2 - Cb2
            v2 = r3j - r3i # Cb2 -Cb1
            #phi[j, i] = torch.acos(torch.dot(v1,v2)/torch.norm(v1)/torch.norm(v2))
            phi[j, i] = torch.dot(v1, v2) / torch.norm(v1) / torch.norm(v2)

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


    return d, omega, phi, theta

def crossProdMat(V1,V2):
    Vcp = torch.zeros(V1.shape)
    Vcp[:,:,0] =  V1[:,:,1]*V2[:,:,2] - V1[:,:,2]*V2[:,:,1]
    Vcp[:,:,1] = -V1[:,:,0]*V2[:,:,2] + V1[:,:,2]*V2[:,:,0];
    Vcp[:,:,2] =  V1[:,:,0]*V2[:,:,1] - V1[:,:,1]*V2[:,:,0];
    return Vcp

def ang2plainMat(v1,v2,v3,v4):
    nA = crossProdMat(v1,v2)
    nB = crossProdMat(v3, v4)
    nA = nA/(torch.sqrt(torch.sum(nA**2,axis=2)).unsqueeze(2))
    nB = nB/(torch.sqrt(torch.sum(nB**2,axis=2)).unsqueeze(2))

    cosPsi = torch.sum(nA*nB,axis=2)
    #Psi    = torch.acos(cosPsi)
    return cosPsi



def convertCoordToDistAnglesVec(rN, rCa, rCb, mask=None):
    # Vectorized

    # Get D
    D = torch.sum(rCb ** 2, dim=1).unsqueeze(1) + torch.sum(rCb ** 2, dim=1).unsqueeze(0) - 2 * (rCb @ rCb.t())
    M = mask.unsqueeze(1) @  mask.unsqueeze(0)
    D = torch.sqrt(torch.relu(M*D))

    # Get Upper Phi
    # TODO clean Phi to be the same as OMEGA
    V1x = rCa[:, 0].unsqueeze(1) - rCb[:, 0].unsqueeze(1)
    V1y = rCa[:, 1].unsqueeze(1) - rCb[:, 1].unsqueeze(1)
    V1z = rCa[:, 2].unsqueeze(1) - rCb[:, 2].unsqueeze(1)
    V2x = rCb[:, 0].unsqueeze(1) - rCb[:, 0].unsqueeze(1).t()
    V2y = rCb[:, 1].unsqueeze(1) - rCb[:, 1].unsqueeze(1).t()
    V2z = rCb[:, 2].unsqueeze(1) - rCb[:, 2].unsqueeze(1).t()
    # Normalize them
    V1n = torch.sqrt(V1x**2 + V1y**2 + V1z**2)
    V1x = V1x/V1n
    V1y = V1y/V1n
    V1z = V1z/V1n
    V2n = torch.sqrt(V2x**2 + V2y**2 + V2z**2)
    V2x = V2x/V2n
    V2y = V2y/V2n
    V2z = V2z/V2n
    # go for it
    PHI = M*(V1x * V2x + V1y * V2y + V1z * V2z)
    indnan = torch.isnan(PHI)
    PHI[indnan] = 0.0

    # Omega
    nat = rCa.shape[0]
    V1 = torch.zeros(nat, nat, 3)
    V2 = torch.zeros(nat, nat, 3)
    V3 = torch.zeros(nat, nat, 3)
    # Ca1 - Cb1
    V1[:,:,0] = (rCa[:,0].unsqueeze(1) - rCb[:,0].unsqueeze(1)).repeat((1,nat))
    V1[:,:,1] = (rCa[:,1].unsqueeze(1) - rCb[:,1].unsqueeze(1)).repeat((1, nat))
    V1[:,:,2] = (rCa[:,2].unsqueeze(1) - rCb[:,2].unsqueeze(1)).repeat((1, nat))
    # Cb1 - Cb2
    V2[:,:,0] = rCb[:,0].unsqueeze(1) - rCb[:,0].unsqueeze(1).t()
    V2[:,:,1] = rCb[:,1].unsqueeze(1) - rCb[:,1].unsqueeze(1).t()
    V2[:,:,2] = rCb[:,2].unsqueeze(1) - rCb[:,2].unsqueeze(1).t()
    # Cb2 - Ca2
    V3[:,:,0] = (rCb[:,0].unsqueeze(0) - rCa[:,0].unsqueeze(0)).repeat((nat,1))
    V3[:,:,1] = (rCb[:,1].unsqueeze(0) - rCa[:,1].unsqueeze(0)).repeat((nat,1))
    V3[:,:,2] = (rCb[:,2].unsqueeze(0) - rCa[:,2].unsqueeze(0)).repeat((nat,1))

    OMEGA     = M*ang2plainMat(V1, V2, V2, V3)
    indnan = torch.isnan(OMEGA)
    OMEGA[indnan] = 0.0

    # Theta
    V1 = torch.zeros(nat, nat, 3)
    V2 = torch.zeros(nat, nat, 3)
    V3 = torch.zeros(nat, nat, 3)
    # N - Ca
    V1[:,:,0] = (rN[:,0].unsqueeze(1) - rCa[:,0].unsqueeze(1)).repeat((1,nat))
    V1[:,:,1] = (rN[:,1].unsqueeze(1) - rCa[:,1].unsqueeze(1)).repeat((1, nat))
    V1[:,:,2] = (rN[:,2].unsqueeze(1) - rCa[:,2].unsqueeze(1)).repeat((1, nat))
    # Ca - Cb # TODO - repeated computation
    V2[:,:,0] = (rCa[:,0].unsqueeze(1) - rCb[:,0].unsqueeze(1)).repeat((1,nat))
    V2[:,:,1] = (rCa[:,1].unsqueeze(1) - rCb[:,1].unsqueeze(1)).repeat((1, nat))
    V2[:,:,2] = (rCa[:,2].unsqueeze(1) - rCb[:,2].unsqueeze(1)).repeat((1, nat))
    # Cb1 - Cb2 # TODO - repeated computation
    V3[:,:,0] = rCb[:,0].unsqueeze(1) - rCb[:,0].unsqueeze(1).t()
    V3[:,:,1] = rCb[:,1].unsqueeze(1) - rCb[:,1].unsqueeze(1).t()
    V3[:,:,2] = rCb[:,2].unsqueeze(1) - rCb[:,2].unsqueeze(1).t()

    THETA = M*ang2plainMat(V1, V2, V2, V3)
    indnan = torch.isnan(THETA)
    THETA[indnan] = 0.0
    M[indnan]     = 0.0
    return D, OMEGA, PHI, THETA, M

# Use the codes to process the data and get X and Y
def getProteinData(seq, pssm2, entropy, RN, RCa, RCb, mask, idx, ncourse):

    L2np = ListToNumpy()

    Sh, S, E, rN, rCa, rCb, msk = L2np(seq[idx], pssm2[idx], entropy[idx],
                                       RN[idx], RCa[idx], RCb[idx], mask[idx])
    S   = torch.tensor(S)
    Sh  = torch.tensor(Sh)
    E   = torch.tensor(E)

    nc = ncourse
    kp = S.shape[0]
    k  = (2**nc)*(kp//(2**nc) + 1)
    X  = torch.zeros(21,k,k)
    for i in range(20):
        X[i,:kp,:kp] = S[:,i].unsqueeze(1)@S[:,i].unsqueeze(1).t()
    X[-1,:kp,:kp] = E.unsqueeze(1)@E.unsqueeze(1).t()
    X = X.unsqueeze(0)

    rN  = torch.tensor(rN)
    rCa = torch.tensor(rCa)
    rCb = torch.tensor(rCb)
    msk = torch.tensor(msk)

    # Define model and data Y
    D, OMEGA, PHI, THETA, M = convertCoordToDistAnglesVec(rN,rCa,rCb,mask=msk)
    Yobs = torch.zeros(4, k, k)
    Yobs[0, :kp, :kp] = D
    Yobs[1, :kp, :kp] = OMEGA
    Yobs[2, :kp, :kp] = PHI
    Yobs[3, :kp, :kp] = THETA

    Yobs = Yobs.unsqueeze(0)

    Mpad = torch.zeros(1, k, k)
    Mpad[0, :kp, :kp] = M

    return X, Yobs, Mpad

def plotProteinData(Y,j):

    plt.figure(j)
    plt.subplot(2,2,1)
    plt.imshow(Y[0,0,:,:])
    plt.subplot(2,2,2)
    plt.imshow(Y[0,1,:,:])
    plt.subplot(2,2,3)
    plt.imshow(Y[0,2,:,:])
    plt.subplot(2,2,4)
    plt.imshow(Y[0,3,:,:])
