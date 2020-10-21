import copy

import numpy as np
from numpy.linalg import norm
import random
import matplotlib.pyplot as plt

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

class ConvertDistAnglesToBins(object):
    def __init__(self, nbins_d=36, nbins_omega=24, nbins_theta=24, nbins_phi=12, remove_angles_from_max_dist=True):
        self.nbins_d = nbins_d
        self.nbins_omega = nbins_omega
        self.nbins_phi = nbins_phi
        self.nbins_theta = nbins_theta
        self.remove_angles = remove_angles_from_max_dist
    def __call__(self, targets):
        d = targets[0]
        omega = targets[1]
        phi = targets[2]
        theta = targets[3]

        bins_d = np.linspace(250, 2000, self.nbins_d)
        bins_omega = np.linspace(15, 360, self.nbins_omega)
        bins_phi = np.linspace(15, 180, self.nbins_phi)
        bins_theta = np.linspace(15, 360, self.nbins_theta)
        mask_unknown = d == 0

        d_cat = np.digitize(d, bins=bins_d)
        omega_cat = np.digitize(omega, bins=bins_omega)
        phi_cat = np.digitize(phi, bins=bins_phi)
        theta_cat = np.digitize(theta, bins=bins_theta)

        # Now we make sure that all the unknown gets set to -100, which is the standard in pytorch for ignored values
        d_cat[mask_unknown] = -100
        omega_cat[mask_unknown] = -100
        phi_cat[mask_unknown] = -100
        theta_cat[mask_unknown] = -100

        if self.remove_angles:
            d_mask = d_cat == self.nbins_d
            omega_cat[d_mask] = self.nbins_omega
            phi_cat[d_mask] = self.nbins_phi
            theta_cat[d_mask] = self.nbins_theta

        return d_cat, omega_cat, phi_cat, theta_cat



class ConvertPnetFeaturesTo2D(object):
    def __init__(self):
        pass
    def __call__(self, features):
        seq_onehot = np.eye(len(AA_DICT), dtype=np.float32)[features[0]]

        f1d = np.concatenate((seq_onehot, features[1], features[2][:, None]), axis=1)

        f2d = np.concatenate(
            [np.tile(f1d[:, None, :], [1, f1d.shape[0], 1]), np.tile(f1d[None, :, :], [f1d.shape[0], 1, 1])],
            axis=-1).transpose((-1, 0, 1))
        return f2d

class ConvertPnetFeaturesTo1D(object):
    """
    This assumes that the first feature needs to be converted to a one_hot sequence, and that the rest just needs to be stacked.
    """
    def __init__(self):
        pass
    def __call__(self, features):
        seq_onehot = np.eye(len(AA_DICT), dtype=np.float32)[features[0]]
        if len(features) == 1:
            f1d = seq_onehot.transpose(1, 0)
        elif len(features) == 2:
            if features[1].ndim == 1:
                f1d = np.concatenate((seq_onehot, features[1][:, None]), axis=1).transpose(1, 0)
            else:
                f1d = np.concatenate((seq_onehot, features[1]), axis=1).transpose(1, 0)
        elif len(features) == 3:
            if features[2].ndim == 1:
                f1d = np.concatenate((seq_onehot, features[1], features[2][:, None]), axis=1).transpose(1, 0)
            else:
                f1d = np.concatenate((seq_onehot, features[1], features[2]), axis=1).transpose(1, 0)
        else:
            raise NotImplementedError("ConvertPnetFeaturesTo1D has not been generalized to handle the amount of features you used.")

        return f1d



class SeqFlip(object):
    '''
    This function is specially designed for flipping the features made in pnet. For other types of features, this might not work.
    '''
    def __init__(self, prob = 1.5):
        self.prob = prob
        self.p = random.random()

    def __call__(self, args):
        if self.p > self.prob:
            new_args = ()
            if isinstance(args[0],list) or isinstance(args[0],np.ndarray):
                for arg in args:
                    if isinstance(arg,list):
                        arg.reverse()
                        new_args += (arg,)
                    elif isinstance(arg,np.ndarray):
                        if arg.ndim == 1:
                            new_args += (np.flip(arg, axis=0),)
                        elif arg.ndim == 2:
                            if arg.shape[0] == arg.shape[1]:
                                new_args += (np.flip(arg, axis=(0, 1)),)  # = np.flip(arg),axis=(0, 1))
                            else:
                                new_args += (np.flip(arg, axis=0),)
                        else:
                            raise NotImplementedError("the array you are attempting to flip does not have an implemented shape")
            else:
                args.reverse()
                new_args += (args,)
            args = new_args
        return args

    def reroll(self):
        self.p = random.random()
        return


    def __repr__(self):
        return self.__class__.__name__ + '()'

class ListToNumpy(object):
    def __init__(self):
        pass
    def __call__(self, args):
        args_array = ()
        for arg in args:
            args_array += (np.asarray(arg),)
        return args_array


class ConvertCoordToDists(object):
    def __init__(self):
        pass

    def __call__(self, args):
        distances = ()
        coords = ()
        for r in args:
            mask = r[0,:] != 0
            M = mask[:,None] @  mask[None,:]
            d = np.sum(r ** 2, axis=0)[:,None] + np.sum(r ** 2, axis=0)[None,:] - 2 * (r.T @ r)
            d = np.sqrt(np.maximum(M*d,0))
            distances += (d,)
            coords += (r,)

        return distances, coords


class DrawFromProbabilityMatrix(object):
    '''
    Given a probability matrix P, we generate a vector where each element in the vector is drawn randomly according to the probabilities in the probability matrix.
    P: ndarray of shape (l,n), where l length of the protein (in amino acids), and n is the number of possible amino acids in the one hot encoding.
    NOTE it is assumed that each row in the probability matrix P sums to 1.
    '''
    def __init__(self,sanity_check=False,fraction_of_seq_drawn=1):
        self.sanity_check = sanity_check
        self.fraction_drawn = fraction_of_seq_drawn

    def __call__(self, P, seq=None, debug=False):
        if self.sanity_check:
            self.run_sanity_check(P)
        if debug:
            self.run_debug(P, seq)
        u = np.random.rand(P.shape[0])
        idxs = (P.cumsum(1) < u[:, None]).sum(1)
        if self.fraction_drawn < 1:
            u2 = np.random.rand(P.shape[0])
            replace = u2 > self.fraction_drawn
            idxs[replace] = seq[replace]
        return idxs

    def run_debug(self,P, seq):
        if seq is not None:
            p = seq
        else:
            p = np.argmax(P,axis=1)
        n = P.shape[0]
        iter = 10000
        counter = np.zeros_like(P)
        ndif = np.zeros(n+1)
        for i in range(iter):
            t = self.__call__(P, seq=seq)
            counter[np.arange(n),t] += 1
            tmp = np.sum(t != p)
            ndif[tmp] += 1

        P_pred = counter/ iter
        dif = np.linalg.norm(P - P_pred)
        print("Difference between cumulated drawn examples, and the probability distribution {:2.4e}".format(dif))
        plt.plot(np.arange(n+1)/n,ndif)
        plt.pause(0.5)
        return

    def run_sanity_check(self,P):
        assert (np.abs(np.sum(P,axis=1) - 1) < 1e-6 ).all()
        return



class MaskRandomSubset(object):
    '''
    '''
    def __init__(self):
        self.max_ratio = 0.3
        pass

    def __call__(self, r):
        m = np.ones(r.shape[1])
        pos_range = np.arange(10,r.shape[1]-10)
        endpoints = np.random.choice(pos_range,size=2,replace=False)
        endpoints = np.sort(endpoints)
        r_m = copy.deepcopy(r)
        r_m[:,endpoints[0]:endpoints[1]] = 0
        idx = r_m[0,:] == 0
        m[idx] = 0
        return r_m, m


def one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])