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


class list2np(object):
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


def torsionAngle(r1,r2,r3,r4,M=1.0):

    a = getPairwiseDiff(r2,r1)
    b = getPairwiseDiff(r3,r2)
    c = getPairwiseDiff(r4,r3)

    a = a/torch.sqrt(dotProdMat(a,a).unsqueeze(0))
    b = b/torch.sqrt(dotProdMat(b,b).unsqueeze(0))
    c = c/torch.sqrt(dotProdMat(c,c).unsqueeze(0))

    bXc = crossProdMat(b, c)
    x = dotProdMat(a, c) + dotProdMat(a, b) * dotProdMat(b, c)
    y = dotProdMat(a, bXc)

    PHI = torch.acos(x/torch.sqrt(x**2 + y**2 + 1e-8))

    # (b.((cxb)x(axb)) , |b|(axb).(cxb)
    #cs    = dotProdMat(b,crossProdMat(crossProdMat(c,b),crossProdMat(a,b)))
    #normb = torch.sqrt(dotProdMat(b,b))
    #sn    = normb * dotProdMat(crossProdMat(a,b),crossProdMat(c,b))
    #psi = torch.atan2(cs,sn)
    return M*PHI

def getPairwiseDiff(rCa,rCb):
    # Getting a matrix of rCa_i - rCb_j
    n = rCa.shape[0]
    m = rCb.shape[0]

    V = torch.zeros(3,n,m)
    V[0,:,:] = rCa[:, 0].unsqueeze(1) - rCb[:, 0].unsqueeze(0)
    V[1,:,:] = rCa[:, 1].unsqueeze(1) - rCb[:, 1].unsqueeze(0)
    V[2,:,:] = rCa[:, 2].unsqueeze(1) - rCb[:, 2].unsqueeze(0)

    return V


def crossProdMat(V1,V2):
    Vcp = torch.zeros(V1.shape)
    Vcp[0,:,:] =  V1[1,:,:]*V2[2,:,:] - V1[2,:,:]*V2[1,:,:]
    Vcp[1,:,:] = -V1[0,:,:]*V2[2,:,:] + V1[2,:,:]*V2[0,:,:]
    Vcp[2,:,:] =  V1[0,:,:]*V2[1,:,:] - V1[1,:,:]*V2[0,:,:]
    return Vcp

def dotProdMat(V1,V2):
    Vdp = torch.sum(V1*V2,dim=0)
    return Vdp

def ang2plainMat(v1,v2,v3,v4):
    nA = crossProdMat(v1,v2)
    nB = crossProdMat(v3, v4)
    nA = nA/(torch.sqrt(torch.sum(nA**2,axis=2)).unsqueeze(2))
    nB = nB/(torch.sqrt(torch.sum(nB**2,axis=2)).unsqueeze(2))

    cosPsi = torch.sum(nA*nB,axis=2)
    #Psi    = torch.acos(cosPsi)
    return cosPsi

def convertCoordToDistMaps(rN, rCa, rCb, mask=None):

    # Central coordinate
    rM = (rN + rCa + rCb)/3
    M = mask.unsqueeze(1) @ mask.unsqueeze(0)
    # Get Dmm
    D = torch.sum(rM**2, dim=1).unsqueeze(1) + torch.sum(rM**2, dim=1).unsqueeze(0) - 2*(rM@rM.t())
    Dmm = torch.sqrt(torch.relu(M * D))

    # Get Dmb
    D = torch.sum(rM**2, dim=1).unsqueeze(1) + torch.sum(rCb**2, dim=1).unsqueeze(0) - 2*(rM@rCb.t())
    Dmb = torch.sqrt(torch.relu(M*D))

    # Get Dma
    D = torch.sum(rM**2, dim=1).unsqueeze(1) + torch.sum(rCa**2, dim=1).unsqueeze(0) - 2 * (rM@rCa.t())
    Dma = torch.sqrt(torch.relu(M*D))

    # Get DmN
    D = torch.sum(rM ** 2, dim=1).unsqueeze(1) + torch.sum(rN ** 2, dim=1).unsqueeze(0) - 2 * (rM @ rN.t())
    DmN = torch.sqrt(torch.relu(M*D))

    return Dmm, Dma, Dmb, DmN, M

def torsionAngleIJ(r1,r2,r3,r4,M=1.0):

    pi = 3.1415926535
    m   = r1.shape[0]
    psi = torch.zeros(m,m, dtype=r1.dtype)
    for i in range(m):
        for j in range(m):
            a = r2[i,:] - r1[j,:]
            b = r3[i,:] - r2[j,:]
            c = r4[i,:] - r3[j,:]
            a = a/torch.norm(a)
            b = b/torch.norm(b)
            c = c/torch.norm(c)
            bXc = torch.cross(b,c)
            x = torch.dot(a,c) + torch.dot(a,b)*torch.dot(b,c)
            y = torch.dot(a, bXc)
            #psi[i,j] = torch.atan2(x,y)
            ang = 0
            if (x != 0) & (y!=0):
                c = x/torch.sqrt(x**2 + y**2)
                ang = torch.sign(y) * torch.acos(c)
            elif(x==0):
                if (y>0):
                    ang = pi/2
                elif(y<0):
                    ang = -pi/2
            psi[i,j] = ang

    return M*psi


def convertCoordToAngles(rN, rCa, rCb, mask=1.0):
    #OMEGA, COMEGA, SOMEGA = torsionAngle(rCa, rCb, rCb, rCa) # Omega: Ca, Cb, Cb, Ca
    #THETA, CTHETA, STHETA = torsionAngle(rN, rCa, rCb, rCb) # N, Ca, Cb, Cb
    #PHI,   CPHI,   SPHI   = torsionAngle(rCb, rCb, rCa, rN) # Cb, Cb, Ca, N
    OMEGA = torsionAngle(rCa, rCb, rCb, rCa)  # Omega: Ca, Cb, Cb, Ca
    THETA = torsionAngle(rN, rCa, rCb, rCb) # N, Ca, Cb, Cb
    PHI   = torsionAngle(rCb, rCb, rCa, rN) # Cb, Cb, Ca, N

    return mask*OMEGA, mask*THETA, mask*PHI



# Use the codes to process the data and get X and Y
def getProteinData(seq, pssm2, entropy, RN, RCa, RCb, mask, idx, ncourse):

    L2np = list2np()

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
    #D, OMEGA, PHI, THETA, M = convertCoordToDistAnglesVec(rN,rCa,rCb,mask=msk)
    Dmm, Dma, Dmb, DmN, M = convertCoordToDistMaps(rN, rCa, rCb, mask=msk)

    Yobs = torch.zeros(4, k, k)
    Yobs[0, :kp, :kp] = Dmm
    Yobs[1, :kp, :kp] = Dma
    Yobs[2, :kp, :kp] = Dmb
    Yobs[3, :kp, :kp] = DmN

    Yobs = Yobs.unsqueeze(0)

    Mpad = torch.zeros(1, k, k)
    Mpad[0, :kp, :kp] = M

    return X, Yobs, Mpad

def plotProteinData(Y,j):

    plt.figure(j)
    plt.subplot(2,2,1)
    plt.imshow(Y[0,0,:,:])
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(Y[0,1,:,:])
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(Y[0,2,:,:])
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(Y[0,3,:,:])
    plt.colorbar()

