import os
import time
import gzip

import lmdb
from xopen import xopen
import numpy as np
import string
import glob

from preprocessing.ANN import ANN_sparse
from preprocessing.MSA_reader import read_a2m_gz_file, sparse_one_hot_encoding, msa2pssm, dca
from preprocessing.pnet2lmdb import dumps_pyarrow
from srcOld.dataloader_utils import AA_DICT, DSSP_DICT, NUM_DIMENSIONS, MASK_DICT, SeqFlip, ListToNumpy, \
    DrawFromProbabilityMatrix
import re
from itertools import compress

from srcOld.dataloader_pnet import read_record, parse_pnet, separate_coords

AA_LIST = 'ACDEFGHIKLMNPQRSTVWY-'


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
    scaling = 0.001  # converts from pico meters to nanometers

    t0 = time.time()
    while True:
        next_line = file_.readline()
        for case in switch(next_line):
            if case('[ID]' + '\n'):
                id.append(file_.readline()[:-1])
                if len(id) % 1000 == 0:
                    print("loading sample: {:}, Time: {:2.2f}".format(len(id), time.time() - t0))
            elif case('[PRIMARY]' + '\n'):
                seq.append(letter_to_num(file_.readline()[:-1], AA_DICT))
                seq_len.append(len(seq[-1]))
            elif case('[EVOLUTIONARY]' + '\n'):
                evolutionary = []
                for residue in range(num_evo_entries):
                    evolutionary.append([float(step) for step in file_.readline().split()])
                # pssm.append(evolutionary)
                entropy_i = [float(step) for step in file_.readline().split()]
                # entropy.append(entropy_i)
            elif case('[SECONDARY]' + '\n'):
                dssp_i = letter_to_num(file_.readline()[:-1], DSSP_DICT)
                # dssp.append(dssp_i)
            elif case('[TERTIARY]' + '\n'):
                tertiary = []
                for axis in range(NUM_DIMENSIONS):
                    tertiary_i = [float(coord) * scaling for coord in file_.readline().split()]
                    tertiary.append(tertiary_i)
                coord.append(tertiary)
            elif case('[MASK]' + '\n'):
                mask_i = letter_to_bool(file_.readline()[:-1], MASK_DICT)
                # mask.append()
            elif case(''):

                return id, seq, pssm, entropy, dssp, coord, mask, seq_len


def parse_pnet_for_comparison(file, min_seq_len=-1, max_seq_len=-1):
    with open(file, 'r') as f:
        t0 = time.time()
        id, seq, pssm, entropy, dssp, coords, mask, seq_len = read_record(f, 20)
        # NOTE THAT THE RESULT IS RETURNED IN ANGSTROM
        print("loading data complete! Took: {:2.2f}".format(time.time() - t0))
        r1 = []
        r2 = []
        r3 = []
        pssm2 = []
        for i in range(len(coords)):  # We transform each of these, since they are inconveniently stored
            #     pssm2.append(flip_multidimensional_list(pssm[i]))
            #     # Note that we are changing the order of the coordinates, as well as which one is first, since we want Carbon alpha to be the first, Carbon beta to be the second and Nitrogen to be the third
            r1.append((separate_coords(coords[i], 1)))
            r2.append((separate_coords(coords[i], 2)))
            r3.append((separate_coords(coords[i], 0)))

        args = (seq, r1,r2,r3,seq_len,id)
        if max_seq_len > 0 or min_seq_len > 0:
            filter = np.full(len(seq), True, dtype=bool)
            for i,seq_i in enumerate(seq):
                if len(seq_i) > max_seq_len or len(seq_i) < min_seq_len:
                    filter[i] = False
            new_args = ()
            for list_i in args:
                new_args += (list(compress(list_i,filter)),)
        else:
            new_args = args


        convert = ListToNumpy()
        seq = convert(new_args[0])
        r1 = convert(new_args[1])
        r2 = convert(new_args[2])
        r3 = convert(new_args[3])
        seq_len = np.array(new_args[4])
        id = new_args[5]

        print("parse complete! Took: {:2.2f}".format(time.time() - t0))
    return seq, seq_len, id, r1, r2, r3


if __name__ == "__main__":
    from pathlib import Path

    pnet_file = "./../data/training_70.pnet"
    max_seq_len = 320
    min_seq_len = 80
    write_freq = 2
    report_freq = 5
    max_samples = 20000
    nsub_size = 8000
    n_repeats = 3


    t0 = time.time()
    seqs, seqs_len, id, r1, r2, r3 = parse_pnet_for_comparison(pnet_file, max_seq_len=max_seq_len, min_seq_len=min_seq_len)

    t1 = time.time()
    print("Read the pnet_file, took {:2.2f}, contains {:} samples".format(t1 - t0, len(seqs_len)))
    seqs_len_unique, counts = np.unique(seqs_len, return_counts=True)
    a = np.digitize(seqs_len, bins=seqs_len_unique, right=True)
    lookup = {}  # create an empty dictionary
    # Make a lookup table such that t[seq_len] = idx
    for i, seq_len_unique in enumerate(seqs_len_unique):
        lookup[seq_len_unique] = i

    # Next we create the list of arrays, and then afterwards we populate them
    seqs_list = []
    seqs_list_org_id = []
    for seq_len_unique, count in zip(seqs_len_unique, counts):
        tmp = np.empty((count, seq_len_unique), dtype=np.int32)
        seqs_list.append(tmp)
        tmp = np.empty((count, 1), dtype=np.int32)
        seqs_list_org_id.append(tmp)

    counter = np.zeros_like(counts)
    for i, (seq, seq_len) in enumerate(zip(seqs, seqs_len)):
        seq_idx = lookup[seq_len]
        counter_idx = counter[seq_idx]
        seqs_list[seq_idx][counter_idx, :] = seq
        seqs_list_org_id[seq_idx][counter_idx] = i
        counter[seq_idx] += 1

    n = len(seqs)
    matches = np.zeros(n, dtype=np.int32)

    # for i,(seq,seq_len) in enumerate(zip(seqs,seqs_len)):
    #     seq_idx = lookup[seq_len]
    #     seqs_i = seqs_list[seq_idx]
    #     res = np.mean(seq == seqs_i,axis=1) == 1
    #     if np.sum(res) == 1:
    #         continue
    #     else:
    #         id = np.where(res == True)[0]
    #         print("{:}, in {:}, {:}".format(i,seq_idx,id))
    #         print("what now?")

    # I think the correct way to do this is to sort pnet, by length of the proteins, and group all proteins with the same length in an nd.array
    # Make a lookup table with idx -> seq_len
    # Then the comparison is only against all elements in that particular group.
    t2 = time.time()
    print("Built the data structure, took {:2.2f}".format(t2 - t1))

    n_matches = 0
    MSA_not_in_protein_net = 0
    MSA_in_protein_net_more_than_once = 0

    MSA_folder = "F:/Globus/raw/"
    outputfolder = "F:/Output2/"
    avg_read_time = 0
    avg_lookup_time = 0
    avg_ann_time = 0
    avg_dca_time = 0
    search_command = MSA_folder + "*.a2m.gz"
    a2mfiles = [f for f in glob.glob(search_command)]

    search_command = outputfolder + "*.npz"
    outputfiles = [f for f in glob.glob(search_command)]
    ite_start = -1
    for outputfile in outputfiles:
        str_tmp = outputfile.split(sep="ID")[-1]
        num = int("".join(filter(str.isdigit, str_tmp)))
        ite_start = max(num,ite_start)
    problems = 0
    t0 = time.time()
    cnt = 0
    n_excluded = 0

    for i, a2mfile in enumerate(a2mfiles):
        if i <= ite_start:
            continue
        # print("{:}".format(i))
        tt0 = time.time()

        msas = read_a2m_gz_file(a2mfile,max_seq_len=max_seq_len,min_seq_len=min_seq_len)

        if msas is None:
            n_excluded += 1
            continue
        tt1 = time.time()
        avg_read_time += tt1 - tt0
        # print("Read: {:2.2f}, Average: {:2.2f}".format(tt1-tt0, avg_read_time/(i+1)))
        protein = msas[-1, :]
        seq_len = len(protein)
        try:
            seq_idx = lookup[seq_len]
        except:
            MSA_not_in_protein_net += 1
            continue
        seqs_i = seqs_list[seq_idx]
        res = np.mean(protein == seqs_i, axis=1) == 1
        tt2 = time.time()
        avg_lookup_time += tt2-tt1
        if np.sum(res) == 1:
            n_matches += 1
            org_id = (seqs_list_org_id[seq_idx][res]).squeeze()
            matches[org_id] += 1
            r1i = r1[org_id].T
            r2i = r2[org_id].T
            r2i = r3[org_id].T
            # print("msa shape {:}, r1 shape {:} ".format(msa.shape,r1i.shape))

            n = msas.shape[0]
            l = msas.shape[1]
            indices = np.random.permutation(n)
            pssms = []
            f2d_dcas = []
            for j in range(max(min(n_repeats, n // max_samples), 1)):
                tt3 = time.time()
                idx = indices[j * max_samples:min((j + 1) * max_samples, n)]
                msa = msas[idx, :]
                x_sp = sparse_one_hot_encoding(msa)

                nsub = min(nsub_size, n)
                w = ANN_sparse(x_sp[0:nsub], x_sp, k=100, eff=300, cutoff=True)
                mask = w == np.min(w)
                wm = np.sum(w[mask])
                if wm > 0.01 * np.sum(
                        w):  # If these account for more than 10% of the total weight, we scale them down to almost 10%.
                    scaling = wm / np.sum(w) / 0.01
                else:
                    scaling = 1
                w[mask] /= scaling

                wn = w / np.sum(w)
                msa_1hot = np.eye(21, dtype=np.float32)[msa]
                tt4 = time.time()
                pssms.append(msa2pssm(msa_1hot, w))
                f2d_dcas.append(dca(msa_1hot, w))
                tt5 = time.time()
                avg_ann_time += tt4-tt3
                avg_dca_time += tt5 - tt4
            f2d_dca = np.mean(np.asarray(f2d_dcas), axis=0)
            pssm = np.mean(np.asarray(pssms), axis=0)

            #SAVE FILE HERE
            fullfileout = "{:}ID_{:}".format(outputfolder,i)
            np.savez(fullfileout,protein=protein,pssm=pssm,dca=f2d_dca,r1=r1[org_id].T,r2=r2[org_id].T,r3=r3[org_id].T)


            cnt += 1


        elif np.sum(res) == 0:
            MSA_not_in_protein_net += 1
        else:
            MSA_in_protein_net_more_than_once += 1
        if (i + 1) % report_freq == 0:
            print("Compared {:} proteins. Matches: {:}, MSA not in pnet: {:}, MSAs in pnet more than once {:}, excluded: {:}, Time(read): {:2.2f}, Time(lookup): {:2.2f}, Time(ANN): {:2.2f}, Time(DCA): {:2.2f}, Time(total): {:2.2f}".format(
                  i + 1, n_matches, MSA_not_in_protein_net, MSA_in_protein_net_more_than_once, n_excluded, avg_read_time/(i+1-n_excluded),avg_lookup_time/(i+1-n_excluded),avg_ann_time/(n_matches+1e-9),avg_dca_time/(n_matches+1e-9),time.time() - t0))

    # finish iterating through dataset
    print("Done: {:}, excluded: {:}, problems: {:} Time: {:2.2f}".format(cnt, n_excluded,
                                                                         problems,
                                                                         time.time() - t0))

print("done")