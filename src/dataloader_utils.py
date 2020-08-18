import numpy as np
from numpy.linalg import norm
import random

# Constants
NUM_DIMENSIONS = 3

# Functions for conversion from Mathematica protein files
AA_DICT = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7', 'K': '8', 'L': '9',
            'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18',
            'Y': '19','-': '20'}
AA_PAD_VALUE = 20
DSSP_DICT = {'L': '0', 'H': '1', 'B': '2', 'E': '3', 'G': '4', 'I': '5', 'T': '6', 'S': '7'}
DSSP_PAD_VALUE = 0 #TODO I DONT KNOW WHETHER THIS IS RIGHT
MASK_DICT = {'-': '0', '+': '1'}
MASK_PAD_VALUE = 0
PSSM_PAD_VALUE = 0
ENTROPY_PAD_VALUE = 0
COORDS_PAD_VALUE = 0

class SeqResizeAndFlip(object):
    def __init__(self, seq_len):
        self.seq_len = seq_len
    def __call__(self, seq, padding_value, idx_start=None,flip=None):
        if flip is None:
            if random.random() > 0.5:
                flip = True
            else:
                flip = False
        if flip:
            seq = seq.reverse()
        n = len(seq)
        if n > self.seq_len:
            if idx_start is None:
                idx_start = random.randint(0,n-self.seq_len)
            seq_new = seq[idx_start:idx_start+self.seq_len]
        else:
            if type(seq[0]) is list:
                inner_list = [padding_value] * (len(seq[0]))
                seq_new = seq + [inner_list] * (self.seq_len-n)
            else:
                seq_new = seq + [padding_value] * (self.seq_len-n)
        return np.asarray(seq_new), idx_start, flip

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SeqFlip(object):
    def __init__(self, prob = 0.5):
        self.prob = prob
    def __call__(self, *args):
        if random.random() > self.prob:
            for arg in args:
                arg.reverse()
        return args

    def __repr__(self):
        return self.__class__.__name__ + '()'

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




def convert_coord_to_dist_angles(coords):
    '''
    Data should be coordinate data in pnet format, meaning that each amino acid is characterized by a 3x3 matrix, which are the coordinates of N,Calpha,Cbeta.
    :param coord:
    :return:
    '''
    dists = []
    phis = []
    omegas = []
    thetas = []
    for coord in coords: #For each protein
        seq_len = len(coord[0])//3

        d = np.zeros([seq_len, seq_len])
        phi = np.zeros([seq_len, seq_len])
        omega = np.zeros([seq_len, seq_len])
        theta = np.zeros([seq_len, seq_len])
        for i in range(seq_len):
            for j in range(seq_len):
                Cbi = np.array([coord[0][i*3+2], coord[1][i*3+2], coord[2][i*3+2]])
                Cbj = np.array([coord[0][j*3+2], coord[1][j*3+2], coord[2][j*3+2]])
                Cai = np.array([coord[0][i*3+1], coord[1][i*3+1], coord[2][i*3+1]])
                Ni = np.array([coord[0][i*3], coord[1][i*3], coord[2][i*3]])
                Nj = np.array([coord[0][j*3], coord[1][j*3], coord[2][j*3]])


                a = norm(Cbi-Cbj)
                b = norm(Cbi-Cai)
                c = norm(Cbj-Cai)

                phi[i,j] = np.degrees(np.arccos((a*a + b*b - c*c)/(2*a*b)))

                Caj = np.array([coord[0][j*3+1], coord[1][j*3+1], coord[2][j*3+1]])
                v1 = Cbj - Cbi
                v2 = Cai - Cbi
                normal_vec = np.cross(v1,v2)
                v3 = Caj - Cbj

                #Now we find thetas
                v4 = Ni - Cai
                v4_proj = proj_3d(v4, Cai - Cbi)
                v4_ort = v4 - v4_proj
                theta[i,j] = (np.degrees(np.arccos(np.dot(v4_ort,normal_vec) / (norm(v4_ort) * norm(normal_vec)))) + 90) % 360

                if i > j: #These two are symmetric so we only calculate half of them
                    d[i,j] = norm(Cbi-Cbj)
                    #First project this vector on the vector running between Cbi Cbj
                    v3_proj = proj_3d(v3,v1)
                    v3_ort = v3 - v3_proj
                    #Now find the angle between the normal vector and project vector and add 90 to make it to the plane
                    omega[i,j] = (np.degrees(np.arccos(np.dot(v3_ort,normal_vec) / (norm(v3_ort) * norm(normal_vec)))) + 90) % 360


        d = d + d.transpose()
        dists.append(d)
        omega = omega + omega.transpose()
        omegas.append(omega)

        mask_nan = np.isnan(phi)
        phi[mask_nan] = 0

        mask_nan = np.isnan(theta)
        theta[mask_nan] = 0

        phis.append(phi)
        thetas.append(theta)

    return dists,omegas,phis,thetas

def one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])