import os
import random
import re
import time

import numpy as np
from torch.utils.data import Dataset

from supervised.dataloader_utils import AA_DICT, DSSP_DICT, NUM_DIMENSIONS, MASK_DICT, SeqFlip, ListToNumpy, \
    DrawFromProbabilityMatrix


class Dataset_pnet(Dataset):
    def __init__(self, file, transform=None, transform_target=None, transform_mask=None, max_seq_len=300, chan_in=21, chan_out=3, draw_seq_from_msa=False):
        args = parse_pnet(file,max_seq_len=max_seq_len, use_entropy=True, use_pssm=True, use_dssp=False, use_mask=True, use_coord=True)
        self.file = file
        self.seq = args['seq']
        self.pssm = args['pssm']
        self.entropy = args['entropy']
        self.mask = args['mask']
        self.r1 = args['r1']  # Ca
        self.r2 = args['r2']  # Cb
        self.r3 = args['r3']  # N

        self.transform = transform
        self.transform_target = transform_target
        self.transform_mask = transform_mask
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.draw_seq_from_msa = draw_seq_from_msa
        self.draw = DrawFromProbabilityMatrix(fraction_of_seq_drawn=0.2)
        self.draw_prob = 0.5
        # self.nfeatures = 84

    def __getitem__(self, index):
        features = self.select_features(index)
        target = (self.r1[index], self.r2[index], self.r3[index])

        target = self.match_target_channels(target)

        if isinstance(self.transform.transforms[0], SeqFlip):
            self.transform.transforms[0].reroll()
        if self.transform is not None:
            features = self.transform(features)
        if self.transform_target is not None:
            distances, coords = self.transform_target(target)

        return features, distances, coords

    def select_features(self,index):
        p = random.random()
        if self.chan_in == 21:
            if self.draw_seq_from_msa and p > self.draw_prob:
                features = (self.draw(self.pssm[index], seq=self.seq[index]),)
            else:
                features = (self.seq[index],)
        elif self.chan_in == 22:
            if self.draw_seq_from_msa and p > self.draw_prob:
                features = (self.draw(self.pssm[index], seq=self.seq[index]), self.entropy[index])
            else:
                features = (self.seq[index], self.entropy[index])
        elif self.chan_in == 41:
            features = (self.seq[index], self.pssm[index])
        elif self.chan_in == 42:
            features = (self.seq[index], self.pssm[index], self.entropy[index])
        else:
            raise NotImplementedError("The selected number of channels in is not currently supported")
        return features

    def match_target_channels(self,target):
        if self.chan_out == 3:
            target = (target[0],)
        elif self.chan_out == 6:
            target = target[0:2]
        elif self.chan_out == 9:
            pass
        else:
            raise NotImplementedError("Chan_out is {}, which is not implemented".format(self.chan_out))
        return target


    def __len__(self):
        return len(self.seq)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.file + ')'

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



def read_record(file_, num_evo_entries, use_entropy, use_pssm, use_dssp, use_mask, use_coord, AA_DICT, report_iter=1000, min_seq_len=-1, max_seq_len=999999,save=False,scaling=1, min_known_ratio=0):
    """
    Read all protein records from pnet file.
    Note that pnet files have coordinates saved in picometers, which is not the normal standard.
    """

    id = []
    seq = []
    pssm = []
    entropy = []
    dssp = []
    coord = []
    mask = []
    seq_len = []

    t0 = time.time()
    cnt = 0
    while True:
        next_line = file_.readline()
        for case in switch(next_line):
            if case('[ID]' + '\n'):
                cnt += 1
                id_i = file_.readline()[:-1]
            elif case('[PRIMARY]' + '\n'):
                seq_i = letter_to_num(file_.readline()[:-1], AA_DICT)
                seq_len_i = len(seq_i)
                if seq_len_i <= max_seq_len and seq_len_i >= min_seq_len:
                    seq_ok = True
                    id.append(id_i)
                    seq.append(seq_i)
                    seq_len.append(seq_len_i)
                else:
                    seq_ok = False
                if (cnt + 1) % report_iter == 0:
                    print("Reading sample: {:}, accepted samples {:} Time: {:2.2f}".format(cnt, len(id), time.time() - t0))
            elif case('[EVOLUTIONARY]' + '\n'):
                evolutionary = []
                for residue in range(num_evo_entries):
                    evolutionary.append([float(step) for step in file_.readline().split()])
                if use_pssm and seq_ok:
                    pssm.append(evolutionary)
                entropy_i = [float(step) for step in file_.readline().split()]
                if use_entropy and seq_ok:
                    entropy.append(entropy_i)
            elif case('[SECONDARY]' + '\n'):
                dssp_i = letter_to_num(file_.readline()[:-1], DSSP_DICT)
                if use_dssp and seq_ok:
                    dssp.append(dssp_i)
            elif case('[TERTIARY]' + '\n'):
                tertiary = []
                for axis in range(NUM_DIMENSIONS):
                    tertiary.append([float(coord)*scaling for coord in file_.readline().split()])
                if use_coord and seq_ok:
                    coord.append(tertiary)
            elif case('[MASK]' + '\n'):
                mask_i = letter_to_bool(file_.readline()[:-1], MASK_DICT)
                if use_mask and seq_ok:
                    mask.append(mask_i)
            elif case(''):
                return id,seq,pssm,entropy,dssp,coord,mask,seq_len

def parse_pnet(file, log_unit=-9, min_seq_len=-1, max_seq_len=999999, use_entropy=True, use_pssm=True, use_dssp=False, use_mask=True, use_coord=True, AA_DICT=AA_DICT):
    """
    This is a wrapper for the read_record routine, which reads a pnet-file into memory.
    This routine will convert the lists to numpy arrays, and flip their dimensions to be consistent with how data is normally used in a deep neural network.
    Furthermore the routine will specify the log_unit you wish the data in default is -9 which is equal to nanometer. (Pnet data is given in picometer = -12 by standard)
    """
    with open(file, 'r') as f:
        pnet_log_unit = -12
        scaling = 10.0 ** (pnet_log_unit - log_unit)
        t0 = time.time()
        id, seq, pssm, entropy, dssp, coords, mask, seq_len = read_record(f, 20, AA_DICT=AA_DICT, use_entropy=use_entropy, use_pssm=use_pssm, use_dssp=use_dssp, use_mask=use_mask, use_coord=use_coord, min_seq_len=min_seq_len, max_seq_len=max_seq_len, scaling=scaling, min_known_ratio=min_known_ratio)
        print("loading data complete! Took: {:2.2f}".format(time.time() - t0))
        rCa = []
        rCb = []
        rN = []

        for i in range(len(coords)):  # We transform each of these, since they are inconveniently stored
            #     # Note that we are changing the order of the coordinates, as well as which one is first, since we want Carbon alpha to be the first, Carbon beta to be the second and Nitrogen to be the third
            rCa.append((separate_coords(coords[i], 1)))
            rCb.append((separate_coords(coords[i], 2)))
            rN.append((separate_coords(coords[i], 0)))

        convert = ListToNumpy()
        seq = convert(seq)
        seq_len = np.array(seq_len)
        args = {'id': id,
                'seq': seq,
                'seq_len': seq_len,
                }
        if use_coord:
            rCa = convert(rCa)
            rCb = convert(rCb)
            rN = convert(rN)

            args['rCa'] = rCa
            args['rCb'] = rCb
            args['rN'] = rN
        if use_entropy:
            entropy = convert(entropy)
            args['entropy'] = entropy
        if use_pssm:
            pssm = convert(pssm)
            args['pssm'] = pssm
        if use_dssp:
            dssp = convert(dssp)
            args['dssp'] = dssp
        if use_mask:
            mask = convert(mask)
            args['mask'] = mask
        print("parsing pnet complete! Took: {:2.2f}".format(time.time() - t0))
    return args, log_unit, AA_DICT

if __name__ == '__main__':
    # pnetfile = './../data/casp11/training_90'
    # output_folder = './../data/casp11_training_90/'
    # pnetfile = './../data/casp11/testing'
    # output_folder = './../data/casp11_testing/'
    pnetfile = './../data/casp11/validation'
    output_folder = './../data/casp11_validation/'

    os.makedirs(output_folder, exist_ok=True)
    args, log_units, AA_DICT = parse_pnet(pnetfile, min_seq_len=-1, max_seq_len=1000, use_entropy=True, use_pssm=True, use_dssp=False, use_mask=False, use_coord=True)

    ids = args['id']
    rCa = args['rCa']
    rCb = args['rCb']
    rN = args['rN']
    pssm = args['pssm']
    entropy = args['entropy']
    seq = args['seq']
    AA_LIST = list(AA_DICT)
    for i,id in enumerate(ids):
        filename = "{:}{:}.npz".format(output_folder,id)
        entropy_i = entropy[i]
        np.savez(file=filename, seq=seq[i],pssm=pssm[i],entropy=entropy_i[None,:],rCa=rCa[i].T,rCb=rCb[i].T,rN=rN[i].T,id=id, log_units=log_units, AA_LIST=AA_LIST)


