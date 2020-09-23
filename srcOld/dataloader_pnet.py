import numpy as np
import random

import re
from torch.utils.data import Dataset
import time
from itertools import compress
import copy
from srcOld.dataloader_utils import AA_DICT, DSSP_DICT, NUM_DIMENSIONS, MASK_DICT, SeqFlip, ListToNumpy, \
    DrawFromProbabilityMatrix


class Dataset_pnet(Dataset):
    def __init__(self, file, transform=None, transform_target=None, transform_mask=None, max_seq_len=300, chan_in=21, chan_out=3, draw_seq_from_msa=False):
        seq,pssm,entropy,dssp,r1,r2,r3,mask = parse_pnet(file,max_seq_len=max_seq_len)
        self.file = file
        self.seq = seq
        self.pssm = pssm
        self.entropy = entropy
        self.dssp = dssp
        self.mask = mask
        self.r1 = r1  # Ca
        self.r2 = r2  # Cb
        self.r3 = r3  # N

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



def read_record(file_, num_evo_entries):
    """ Read all protein records from pnet file. """
    id = []
    seq = []
    pssm = []
    entropy = []
    dssp = []
    coord = []
    mask = []
    seq_len = []
    scaling = 0.001 # converts from pico meters to nanometers

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
                seq_len.append(len(seq[-1]))
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
                for axis in range(NUM_DIMENSIONS): tertiary.append([float(coord)*scaling for coord in file_.readline().split()])
                coord.append(tertiary)
            elif case('[MASK]' + '\n'):
                mask.append(letter_to_bool(file_.readline()[:-1], MASK_DICT))
            elif case(''):


                return id,seq,pssm,entropy,dssp,coord,mask,seq_len

def parse_pnet(file,max_seq_len=-1):
    with open(file, 'r') as f:
        t0 = time.time()
        id, seq, pssm, entropy, dssp, coords, mask, seq_len = read_record(f, 20)
        #NOTE THAT THE RESULT IS RETURNED IN ANGSTROM
        print("loading data complete! Took: {:2.2f}".format(time.time()-t0))
        r1 = []
        r2 = []
        r3 = []
        pssm2 = []
        coords = coords
        for i in range(len(pssm)): #We transform each of these, since they are inconveniently stored
            pssm2.append(flip_multidimensional_list(pssm[i]))
            # Note that we are changing the order of the coordinates, as well as which one is first, since we want Carbon alpha to be the first, Carbon beta to be the second and Nitrogen to be the third
            r1.append(flip_multidimensional_list(separate_coords(coords[i], 1)))
            r2.append(flip_multidimensional_list(separate_coords(coords[i], 2)))
            r3.append(flip_multidimensional_list(separate_coords(coords[i], 0)))

            if i+1 % 1000 == 0:
                print("flipping and separating: {:}, Time: {:2.2f}".format(len(id), time.time() - t0))

        args = (seq, pssm2, entropy, dssp, r1,r2,r3, mask, seq_len)
        if max_seq_len > 0:
            filter = np.full(len(seq), True, dtype=bool)
            for i,seq_i in enumerate(seq):
                if len(seq_i) > max_seq_len:
                    filter[i] = False
            new_args = ()
            for list_i in (seq, pssm2, entropy, dssp, r1,r2,r3, mask, seq_len):
                new_args += (list(compress(list_i,filter)),)
        else:
            new_args = args
        convert = ListToNumpy()
        seq = convert(new_args[0])
        r1 = convert(new_args[4])
        r2 = convert(new_args[5])
        r3 = convert(new_args[6])
        seq_len = np.array(new_args[-1])
        new_args = (seq, r1, r2, r3,seq_len)

        print("parse complete! Took: {:2.2f}".format(time.time() - t0))
    return new_args

