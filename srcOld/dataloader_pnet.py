import numpy as np
import re
from torch.utils.data import Dataset
import time
from itertools import compress
import copy
from srcOld.dataloader_utils import AA_DICT, DSSP_DICT, NUM_DIMENSIONS, MASK_DICT


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
        # self.nfeatures = 84

    def __getitem__(self, index):
        features = (self.seq[index], self.pssm[index], self.entropy[index])
        mask = self.mask[index]
        target = (self.r1[index], self.r2[index], self.r3[index])

        self.transform.transforms[0].reroll()
        if self.transform is not None:
            features = self.transform(features)
        if self.transform_target is not None:
            distances, coords = self.transform_target(target)
        # if self.transform_mask is not None:
        #     mask = self.transform_mask(mask) #TODO CHECK THAT THIS IS NOT DOUBLE FLIPPED!

        return features, distances, coords

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


                return id,seq,pssm,entropy,dssp,coord,mask

def parse_pnet(file,max_seq_len=-1):
    with open(file, 'r') as f:
        t0 = time.time()
        id, seq, pssm, entropy, dssp, coords, mask = read_record(f, 20)
        #NOTE THAT THE RESULT IS RETURNED IN ANGSTROM
        print("loading data complete! Took: {:2.2f}".format(time.time()-t0))
        r1 = []
        r2 = []
        r3 = []
        pssm2 = []
        coords = coords
        for i in range(len(pssm)): #We transform each of these, since they are inconveniently stored
            pssm2.append(flip_multidimensional_list(pssm[i]))
            r1.append(flip_multidimensional_list(separate_coords(coords[i], 0)))
            r2.append(flip_multidimensional_list(separate_coords(coords[i], 1)))
            r3.append(flip_multidimensional_list(separate_coords(coords[i], 2)))

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

