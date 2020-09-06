import os
import lmdb
import pyarrow as pa
import re
import time
import numpy as np

from src.dataloader_utils import ListToNumpy, AA_DICT, DSSP_DICT, NUM_DIMENSIONS, MASK_DICT, ConvertCoordToDistAnglesVec
from srcOld.dataloader_utils import ConvertCoordToDists


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


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



def read_pnet_into_lmdb(pnet_file, lmdb_file,max_seq_len=300,min_seq_len=80,num_evo_entries=20,db_size=1e10,write_freq=5000,report_freq=5000):
    """ Read all protein records from pnet file. """

    isdir = os.path.isdir(lmdb_file)

    print("Generate LMDB to %s" % lmdb_file)
    db = lmdb.open(lmdb_file, subdir=isdir,
                   map_size=int(db_size), readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    problems = 0
    t0 = time.time()
    cnt = 0
    n_excluded = 0
    with open(pnet_file, 'r') as f:
        while True:
            next_line = f.readline()
            for case in switch(next_line):
                if case('[ID]' + '\n'):
                    id = f.readline()[:-1]
                elif case('[PRIMARY]' + '\n'):
                    seq = letter_to_num(f.readline()[:-1], AA_DICT)
                elif case('[EVOLUTIONARY]' + '\n'):
                    evolutionary = []
                    for residue in range(num_evo_entries):
                        evolutionary.append([float(step) for step in f.readline().split()])
                    pssm = evolutionary
                    entropy = [float(step) for step in f.readline().split()]
                elif case('[SECONDARY]' + '\n'):
                    dssp = letter_to_num(f.readline()[:-1], DSSP_DICT)
                elif case('[TERTIARY]' + '\n'):
                    tertiary = []
                    for axis in range(NUM_DIMENSIONS): tertiary.append([float(coord) for coord in f.readline().split()])
                    coord = tertiary
                elif case('[MASK]' + '\n'):
                    mask = letter_to_bool(f.readline()[:-1], MASK_DICT)
                    if max_seq_len < len(seq) or min_seq_len > len(seq):
                        n_excluded += 1
                        continue

                    features, target, mask = process_data(seq, pssm, entropy, coord, mask)

                    if np.sum(mask[0]) == 0 or np.sum(target[0]) == 0:
                        problems += 1
                        n_excluded += 1
                        continue


                    txn.put(u'{}'.format(cnt).encode('ascii'), dumps_pyarrow((features, target, mask)))
                    if (cnt+1) % write_freq == 0:
                        print("[%d]" % (cnt+1))
                        txn.commit()
                        txn = db.begin(write=True)
                    if (cnt+1) % report_freq == 0:
                        print("loading sample: {:}, excluded: {:}, problems: {:} Time: {:2.2f}".format(cnt+1,n_excluded, problems, time.time() - t0))
                    cnt += 1
                elif case(''):
                    # finish iterating through dataset
                    txn.commit()
                    keys = [u'{}'.format(k).encode('ascii') for k in range(cnt)]
                    with db.begin(write=True) as txn:
                        txn.put(b'__keys__', dumps_pyarrow(keys))
                        txn.put(b'__len__', dumps_pyarrow(len(keys)))

                    print("Flushing database ...")
                    db.sync()
                    db.close()
                    print("Done: {:}, excluded: {:}, problems: {:} Time: {:2.2f}".format(cnt + 1, n_excluded,
                                                                                                   problems,
                                                                                                   time.time() - t0))
                    return

def process_data(seq,pssm,entropy,coord,mask):
    pssm = flip_multidimensional_list(pssm)
    r1 = separate_coords(coord, 0)
    r2 = separate_coords(coord, 1)
    r3 = separate_coords(coord, 2)
    ltn = ListToNumpy()
    convert = ConvertCoordToDists()
    seq, pssm, entropy, mask, r1, r2, r3 = ltn(seq, pssm, entropy, mask, r1, r2, r3)

    dist = convert(r1, r2, r3, mask)

    target = (dist,)
    features = (seq, pssm, entropy)

    return features, target, (mask,)







if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("-f", "--folder", type=str, default='./../data/training_100.pnet')
    # parser.add_argument("-f", "--folder", type=str, default='./../data/training_30.pnet')
    # parser.add_argument('-s', '--split', type=str, default="val")
    # parser.add_argument('--out', type=str, default="e:/test_lmdb")
    parser.add_argument("-f", "--folder", type=str, default='./../data/testing.pnet')
    parser.add_argument('--out', type=str, default="e:/testing")
    parser.add_argument('-p', '--procs', type=int, default=0)
    max_seq_len = 320
    min_seq_len = 80
    args = parser.parse_args()
    lmdb_name = "{:}_{:}_{:}.lmdb".format(args.out,min_seq_len,max_seq_len)
    read_pnet_into_lmdb(args.folder, lmdb_name, min_seq_len=min_seq_len, max_seq_len=max_seq_len, db_size=1e8, report_freq=1000, write_freq=5000)