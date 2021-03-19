import os
import random
import re
import time
from datetime import datetime

import numpy as np
import torch

# Constants
NUM_DIMENSIONS = 3

# Functions for conversion from Mathematica protein files
AA_DICT = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7', 'K': '8', 'L': '9',
            'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18',
            'Y': '19','-': '20'}

AA_LIST = list(AA_DICT)
# AA_LIST = 'ACDEFGHIKLMNPQRSTVWY-'

AA_PAD_VALUE = 20
DSSP_DICT = {'L': '0', 'H': '1', 'B': '2', 'E': '3', 'G': '4', 'I': '5', 'T': '6', 'S': '7'}
DSSP_PAD_VALUE = 0 #TODO I DONT KNOW WHETHER THIS IS RIGHT
MASK_DICT = {'-': '0', '+': '1'}
MASK_PAD_VALUE = 0
PSSM_PAD_VALUE = 0
ENTROPY_PAD_VALUE = 0
COORDS_PAD_VALUE = 0

class ListToNumpy(object):
    def __init__(self):
        pass
    def __call__(self, args):
        args_array = ()
        for arg in args:
            args_array += (np.asarray(arg),)
        return args_array




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

def parse_pnet_fast(file, log_unit=-9, min_seq_len=-1, max_seq_len=999999, use_entropy=True, use_pssm=True, use_dssp=False,
               use_mask=True, use_coord=True, AA_DICT=AA_DICT, min_ratio=0, sanitize_data=True,
               remove_subproteins=True):
    """
    This is a wrapper for the read_record routine, which reads a pnet-file into memory.
    This routine will convert the lists to numpy arrays, and flip their dimensions to be consistent with how data is normally used in a deep neural network.
    Furthermore the routine will specify the log_unit you wish the data in default is -9 which is equal to nanometer. (Pnet data is given in picometer = -12 by standard)
    """

    def sort_samples_according_to_idx(sort_idx, use_entropy, use_pssm, use_dssp, use_mask, use_coord, rCa, rCb, rN, seq,
                                      seq_len, id, entropy, dssp, pssm, mask):
        seq_len = seq_len[sort_idx]
        seq = [seq[i] for i in sort_idx]
        id = [id[i] for i in sort_idx]

        if use_coord:
            rCa = [rCa[i] for i in sort_idx]
            rCb = [rCb[i] for i in sort_idx]
            rN = [rN[i] for i in sort_idx]
        if use_entropy:
            entropy = [entropy[i] for i in sort_idx]
        if use_pssm:
            pssm = [pssm[i] for i in sort_idx]
        if use_dssp:
            dssp = [dssp[i] for i in sort_idx]
        if use_mask:
            mask = [mask[i] for i in sort_idx]
        return rCa, rCb, rN, seq, seq_len, id, entropy, dssp, pssm, mask

    def keep_samples_according_to_idx(idx_to_keep, use_entropy, use_pssm, use_dssp, use_mask, use_coord, rCa, rCb, rN,
                                      seq, seq_len, id, entropy, dssp, pssm, mask):
        indices = np.where(idx_to_keep == True)[0]

        seq = [seq[index] for index in indices]
        seq_len = seq_len[indices]
        id = [id[index] for index in indices]

        if use_coord:
            rCa = [rCa[index] for index in indices]
            rCb = [rCb[index] for index in indices]
            rN = [rN[index] for index in indices]
        if use_entropy:
            entropy = [entropy[index] for index in indices]
        if use_pssm:
            pssm = [pssm[index] for index in indices]
        if use_dssp:
            dssp = [dssp[index] for index in indices]
        if use_mask:
            mask = [mask[index] for index in indices]
        return rCa, rCb, rN, seq, seq_len, id, entropy, dssp, pssm, mask

    def array_in(arr, sub_arr):
        """
        We wish to do subarray comparison in numpy.
        Let arr be an array of length n
        and subarr an array of length k < n
        We compare to see whether subarr exist contigious anywhere in arr
        """
        n = len(arr)
        k = len(sub_arr)
        idx = np.arange(n - k + 1)[:, None] + np.arange(k)

        comparison = (arr[idx] == sub_arr).all(axis=1)
        result = comparison.any()
        if result:
            idx = np.where(comparison == True)[0][0]
        else:
            idx = -1
        return result, idx

    def array_in_batched(arr, sub_arr):
        """
        We wish to do subarray comparison in numpy.
        Let arr be an array of size ns,l, where ns is number of samples, and l is length
        and subarr an array of length k < l
        We compare to see whether subarr exist contigious anywhere in arr
        """
        ns, l = arr.shape
        k = len(sub_arr)
        idx = np.arange(l - k + 1)[:, None] + np.arange(k)

        comparison = (arr[:, idx] == sub_arr).all(axis=2)
        result = comparison.any()
        if result:
            tmp = np.where(comparison == True)
            samp = tmp[0][0]
            idx = tmp[1][0]
        else:
            samp = -1
            idx = -1
        return result, samp, idx

    assert use_coord == sanitize_data, "Sanitizing data requires coordinate information"
    save_parents = False
    plot_sub_protein_comparison = False
    save_subproteins = False
    min_acceptable_nn_dist = 200  # picometer
    max_acceptable_nn_dist = 1000  # picometer
    min_acceptable_nn_dist_problems = 0
    max_acceptable_nn_dist_problems = 0
    ratio_problems = 0
    pnet_log_unit = -12
    scaling = 10.0 ** (pnet_log_unit - log_unit)
    min_acceptable_nn_dist *= scaling
    max_acceptable_nn_dist *= scaling
    AA_LIST = list(AA_DICT)
    t0 = time.time()
    with open(file, 'r') as f:
        id, seq, pssm, entropy, dssp, coords, mask, seq_len = read_record(f, 20, AA_DICT=AA_DICT,
                                                                          use_entropy=use_entropy, use_pssm=use_pssm,
                                                                          use_dssp=use_dssp, use_mask=use_mask,
                                                                          use_coord=use_coord, min_seq_len=min_seq_len,
                                                                          max_seq_len=max_seq_len, scaling=scaling)
    print("loading data complete! Took: {:2.2f}".format(time.time() - t0))
    n_org = len(seq)
    rCa = []
    rCb = []
    rN = []

    for i in range(len(coords)):  # We transform each of these, since they are inconveniently stored
        #     # Note that we are changing the order of the coordinates, as well as which one is first, since we want Carbon alpha to be the first, Carbon beta to be the second and Nitrogen to be the third
        rCa.append((separate_coords(coords[i], 1)))
        rCb.append((separate_coords(coords[i], 2)))
        rN.append((separate_coords(coords[i], 0)))
    convert = ListToNumpy()
    rCa = convert(rCa)
    rCb = convert(rCb)
    rN = convert(rN)
    seq = convert(seq)
    entropy = convert(entropy)
    pssm = convert(pssm)
    dssp = convert(dssp)
    mask = convert(mask)

    seq_len = np.array(seq_len)

    sort_idx = np.argsort(seq_len)

    rCa, rCb, rN, seq, seq_len, id, entropy, dssp, pssm, mask = sort_samples_according_to_idx(sort_idx, use_entropy,
                                                                                              use_pssm, use_dssp,
                                                                                              use_mask, use_coord, rCa,
                                                                                              rCb, rN, seq, seq_len, id,
                                                                                              entropy, dssp, pssm, mask)

    if min_ratio > 0:
        assert use_coord, "coordinates are required to have min_ratio larger than zero"
        idx_to_keep = np.ones(len(seq), dtype=bool)
        for i in range(len(rCa)):
            n = rCa[i].shape[0]
            m = np.sum(rCa[i][:, 0] != 0)
            ratio = m / n
            if ratio < min_ratio:
                ratio_problems += 1
            idx_to_keep[i] = ratio >= min_ratio
        rCa, rCb, rN, seq, seq_len, id, entropy, dssp, pssm, mask = keep_samples_according_to_idx(idx_to_keep,
                                                                                                  use_entropy, use_pssm,
                                                                                                  use_dssp, use_mask,
                                                                                                  use_coord, rCa, rCb,
                                                                                                  rN, seq, seq_len, id,
                                                                                                  entropy, dssp, pssm,
                                                                                                  mask)

    if sanitize_data:
        # We want to remove suspicious datapoints, that include datapoints where the same coordinate have been repeat multiple times, or where datapoints are suspiciously close or far away from their neighbours
        idx_to_keep = np.ones(len(seq), dtype=bool)
        for i in range(len(rCa)):
            if idx_to_keep[i]:
                rCai = rCa[i]
                m = (rCai[:, 0] != 0).astype(np.float32)
                m2 = np.floor((m[1:] + m[:-1]) / 2.0) < 0.5
                m3 = ~ m2
                drCai = rCai[1:, :] - rCai[:-1, :]
                d = np.sqrt(np.sum(drCai ** 2, axis=1))
                dmin = np.min(d[m3])
                dmax = np.max(d[m3])
                if dmin < min_acceptable_nn_dist:
                    min_acceptable_nn_dist_problems += 1
                    idx_to_keep[i] = False
                if dmax > max_acceptable_nn_dist:
                    max_acceptable_nn_dist_problems += 1
                    idx_to_keep[i] = False
        rCa, rCb, rN, seq, seq_len, id, entropy, dssp, pssm, mask = keep_samples_according_to_idx(idx_to_keep,
                                                                                                  use_entropy, use_pssm,
                                                                                                  use_dssp, use_mask,
                                                                                                  use_coord, rCa, rCb,
                                                                                                  rN, seq, seq_len, id,
                                                                                                  entropy, dssp, pssm,
                                                                                                  mask)

    if remove_subproteins:
        # Note we assume that the proteins have been sorted
        tt0 = time.time()
        n = len(seq)
        idx_to_keep = np.ones(n, dtype=bool)
        parent_proteins = [[] for _ in range(n)]

        seqMat = - np.ones((n,seq_len[-1]),dtype=np.int32)
        for i in range(n):
            seqMat[i,:seq_len[i]] = seq[i]

        for i in range(n):
            if (i + 1) % 1000 == 0:
                print("{:} examples took {:2.2f}".format(i + 1, time.time() - tt0))
            seqi = seq[i]
            result, j, idx = array_in_batched(seqMat[i+1:,:],seqi)
            if result:
                idx_to_keep[i] = False
                parent_proteins[j].append([idx, idx + len(seqi)])
                break

        if save_parents:
            for i in range(n):
                if len(parent_proteins[i]) > 0:
                    sub_proteins = np.asarray(parent_proteins[i])
                    idi = id[i]
                    filename = "{:}parent_protein_{:}.npz".format("./../data/temp/", idi)
                    np.savez(file=filename, seq=seq[i], rCa=rCa[i].T, rCb=rCb[i].T, rN=rN[i].T, id=id,
                             log_units=log_unit, AA_LIST=AA_LIST, subproteins=sub_proteins)

        rCa, rCb, rN, seq, seq_len, id, entropy, dssp, pssm, mask = keep_samples_according_to_idx(idx_to_keep,
                                                                                                  use_entropy, use_pssm,
                                                                                                  use_dssp, use_mask,
                                                                                                  use_coord, rCa, rCb,
                                                                                                  rN, seq, seq_len, id,
                                                                                                  entropy, dssp, pssm,
                                                                                                  mask)

    n_removed = n_org - len(seq)
    args = {'id': id,
            'seq': seq,
            'seq_len': seq_len,
            }
    if use_coord:
        args['rCa'] = rCa
        args['rCb'] = rCb
        args['rN'] = rN
    if use_entropy:
        args['entropy'] = entropy
    if use_pssm:
        args['pssm'] = pssm
    if use_dssp:
        args['dssp'] = dssp
    if use_mask:
        args['mask'] = mask

    print("parsing pnet complete! Took: {:2.2f}".format(time.time() - t0))

    return args, log_unit, AA_DICT, n_removed, min_acceptable_nn_dist_problems, max_acceptable_nn_dist_problems, ratio_problems


if __name__ == '__main__':
    pnetfile = './casp11/testing'
    # output_folder = './../data/casp11_training_90_fully_mapped_no_sub_20_v2/'
    # pnetfile = './../../../data/casp11/validation'
    output_folder = './testing/'
    min_seq_len = 20
    max_seq_len = 600
    use_entropy = True
    use_pssm = True
    use_dssp = False
    use_mask = False
    use_coord = True
    min_ratio = 0.7
    sanitize_data = True

    os.makedirs(output_folder, exist_ok=True)
    args, log_units, AA_DICT, n_removed, min_acceptable_nn_dist_problems, max_acceptable_nn_dist_problems, ratio_problems = parse_pnet_fast(pnetfile, min_seq_len=min_seq_len, max_seq_len=max_seq_len, use_entropy=use_entropy, use_pssm=use_pssm, use_dssp=use_dssp, use_mask=use_mask, use_coord=use_coord, min_ratio=min_ratio, sanitize_data=sanitize_data)

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
    my_file = open("{:}logfile.txt".format(output_folder), "w+")
    my_file.write("Current time = {date:%Y-%m-%d_%H_%M_%S} \n".format(date=datetime.now()))
    my_file.write("pnetfile = {:} \n".format(pnetfile))
    my_file.write("output_folder = {:} \n".format(output_folder))
    my_file.write("min_seq_len = {:} \n".format(min_seq_len))
    my_file.write("max_seq_len = {:} \n".format(max_seq_len))
    my_file.write("use_entropy = {:} \n".format(use_entropy))
    my_file.write("use_pssm = {:} \n".format(use_pssm))
    my_file.write("use_dssp = {:} \n".format(use_dssp))
    my_file.write("use_mask = {:} \n".format(use_mask))
    my_file.write("use_coord = {:} \n".format(use_coord))
    my_file.write("min_ratio = {:} \n".format(min_ratio))
    my_file.write("number of samples saved = {:} \n".format(len(seq)))
    my_file.write("number of samples removed = {:} \n".format(n_removed))
    my_file.write("samples that violated the minimum neighbouring distance = {:} \n".format(min_acceptable_nn_dist_problems))
    my_file.write("samples that violated the maximum neighbouring distance = {:} \n".format(max_acceptable_nn_dist_problems))
    my_file.write("samples that violated the ratio of known coordinates = {:} \n".format(ratio_problems))

