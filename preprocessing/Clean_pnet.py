"""
This file takes the protein net database and cleans it.
Cleaning involves length and whether sufficient info is known
"""
import os

import numpy as np
import time

from srcOld.dataloader_pnet import read_record, switch, letter_to_num, letter_to_bool, flip_multidimensional_list, \
    separate_coords
from srcOld.dataloader_utils import AA_DICT, DSSP_DICT, NUM_DIMENSIONS, MASK_DICT, ListToNumpy


def read_and_clean_pnet(pnet_file, output_folder,max_seq_len=320,min_seq_len=80,num_evo_entries=20,min_ratio=0):
    """ Read all protein records from pnet file. """

    os.makedirs(output_folder)
    scaling = 0.001 # Convert from picometer to nanometer
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
                    for axis in range(NUM_DIMENSIONS): tertiary.append([float(coord)*scaling for coord in f.readline().split()])
                    coord = tertiary
                elif case('[MASK]' + '\n'):
                    mask = letter_to_bool(f.readline()[:-1], MASK_DICT)
                    if max_seq_len < len(seq) or min_seq_len > len(seq):
                        n_excluded += 1
                        continue

                    seq, pssm, entropy, r1, r2, r3, mask = process_data(seq, pssm, entropy, coord, mask)

                    m = r1[0,:] != 0
                    if np.sum(m)/len(m) < min_ratio:
                        problems += 1
                        n_excluded += 1
                        continue
                    #Save data
                    fullfileout = "{:}/{:}".format(output_folder, id)
                    np.savez(fullfileout, seq=seq, pssm=pssm, entropy=entropy, r1=r1,r2=r2,r3=r3, mask=mask)

                    cnt += 1
                elif case(''):
                    print("Done: {:}, excluded: {:}, problems: {:} Time: {:2.2f}".format(cnt, n_excluded,
                                                                                                   problems,
                                                                                                   time.time() - t0))
                    return

def process_data(seq,pssm,entropy,coord,mask):
    pssm = flip_multidimensional_list(pssm)
    r1 = separate_coords(coord, 1) #Ca
    r2 = separate_coords(coord, 2) #Cb
    r3 = separate_coords(coord, 0) #N
    ltn = ListToNumpy()
    seq, pssm, entropy, mask, r1, r2, r3 = ltn((seq, pssm, entropy, mask, r1, r2, r3))

    return seq, pssm, entropy, r1.T, r2.T, r3.T, mask




if __name__ == "__main__":
    pnet_file = "./../data/training_30.pnet"
    # pnet_file = "./../data/testing.pnet"
    max_seq_len = 320
    min_seq_len = 80
    min_ratio_coords_known = 0.7
    output_folder = './../data/clean_pnet_train'
    # output_folder = './../data/clean_pnet_test'

    read_and_clean_pnet(pnet_file, output_folder, max_seq_len=max_seq_len, min_seq_len=min_seq_len, min_ratio=min_ratio_coords_known)

