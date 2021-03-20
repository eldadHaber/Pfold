import glob
import os
import time

import torch

import numpy as np
from pathlib import Path
from Bio.PDB import *
from Bio.PDB.DSSP import DSSP as DSSP_fnc
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SubsMat.MatrixInfo import blosum62
from Bio.pairwise2 import format_alignment

AA_shortener = {
    'ALA': 'A',
    'CYS': 'C',
    'ASP': 'D',
    'GLU': 'E',
    'PHE': 'F',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LYS': 'K',
    'LEU': 'L',
    'MET': 'M',
    'ASN': 'N',
    'PRO': 'P',
    'GLN': 'Q',
    'ARG': 'R',
    'SER': 'S',
    'THR': 'T',
    'VAL': 'V',
    'TRP': 'W',
    'TYR': 'Y',
}

AA_alphabet = 'ACDEFGHIKLMNPQRSTVWY-'


class AA_converter(object):
    def __init__(self, aa_list, unk_idx, unk_tok='-'):
        self.aa_list = aa_list
        self.tok_to_idx = {tok: i for i, tok in enumerate(self.aa_list)}
        self.idx_to_tok = {v: k for k, v in self.tok_to_idx.items()}

        self.unk_idx = unk_idx
        self.unk_tok = unk_tok

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)
    def __call__(self, raw_batch):
        batch_size = len(raw_batch)
        max_len = max(len(seq_str) for seq_str in raw_batch)
        seqs = torch.empty((batch_size, max_len), dtype=torch.int64)
        for i, (seq_str) in enumerate(raw_batch):
            seq = torch.tensor([self.get_idx(s) for s in seq_str], dtype=torch.int64)
            seqs[i, :] = seq
        return seqs

    def reverse(self, seq_num):
        #This assumes a single sequence
        l = seq_num.shape[-1]
        seq_letters = []
        for i in range(l):
            seq_letters.append(self.idx_to_tok.get(seq_num[i],self.unk_tok))
        seq_l = ''.join(seq_letters)
        return seq_l

def match_arrays(array_main, array_sub, exact=False):
    """
    We wish to do subarray comparison in numpy.
    Let arr be an array of length n
    and subarr an array of length k < n
    We compare to see whether subarr exist contigious anywhere in arr
    """
    n = len(array_main)
    k = len(array_sub)
    idx = np.arange(n - k+1)[:, None] + np.arange(k)
    if exact:
        comparison = (array_main[idx] == array_sub).all(axis=1)
        result = comparison.any()
        if result:
            idx = np.where(comparison == True)[0][0]
            score = k
        else:
            idx = -1
            score=0
    else:
        comparison = np.sum((array_main[idx] == array_sub),axis=1)
        score = np.max(comparison)
        idx = np.argmax(comparison)
    return score,idx

def remove_leading_and_trailing_values(seq,val):
    n = seq.shape[-1]
    first = np.where(seq != val)[0][0]
    last = np.where(seq[::-1] != val)[0][0]
    seq_new = seq[first:n-last]
    return seq_new




def find_chain_fit(models_cd,seq,AA_shortener,AA_convert):
    score_best = -1
    idx_best = -1
    # idx1_best = -1
    chain_id_best = -1
    seq_pdb_best = -1
    for model_cd in models_cd.items():
        aa = model_cd[1].child_list
        seq_pdb_long = [AA_shortener.get(i.resname, '-') for i in aa]
        seq_pdb = AA_convert(seq_pdb_long).squeeze().numpy()
        if np.sum(seq_pdb != 20) == 0:
            continue
        seq_pdb = remove_leading_and_trailing_values(seq_pdb,20)
        if len(seq_pdb) > len(seq):
            array_sub = seq
            array_main = seq_pdb
        else:
            array_sub = seq_pdb
            array_main = seq
        score, idx = match_arrays(array_main, array_sub)
        if score > score_best:
            score_best = score
            idx_best = idx
            chain_id_best = model_cd[0]
            seq_pdb_best = seq_pdb
    return idx_best, chain_id_best, seq_pdb_best



def DSSP_wrapper(file,DSSP_ALPHABET,seq,chain_number,chain_id):
    parser = MMCIFParser()
    structure = parser.get_structure("1MOT", file)
    models = structure[0]

    DSSP_results = DSSP_fnc(models, file, dssp='mkdssp')
    a_keys = list(DSSP_results.keys())

    models_cd = models.child_dict
    AA_convert = AA_converter(AA_alphabet, 20)

    if chain_id == -1:
        idx, chain_id, seq_pdb = find_chain_fit(models_cd, seq, AA_shortener, AA_convert)


    #
    # ii = [a_key[1][1] for a_key in a_keys]
    # ii = np.asarray(ii)
    # aa = (ii == 1).nonzero()[0]
    # bb = np.empty(len(aa),dtype=np.int64)
    # bb[:-1] = aa[1:] - aa[:-1]
    # bb[-1] = len(a_keys) - aa[-1]

    chain_ids = [a_key[0] for a_key in a_keys]
    to_use = np.asarray([i == chain_id for i in chain_ids])

    unk_idx = -1
    convert = AA_converter(DSSP_ALPHABET, unk_idx)

    dssp = []
    seq_dssp = []
    for a_key,use_i in zip(a_keys,to_use):
        if use_i:
            tmp = DSSP_results[a_key]
            seq_dssp.append(tmp[1])
            dssp.append(tmp[2])
    dssp_num = convert(dssp)
    seq_dssp = ''.join(seq_dssp)
    seq_dssp_num = AA_convert(seq_dssp).squeeze().numpy()
    seq_l = AA_convert.reverse(seq)
    # seq_pdb_l = AA_convert.reverse(seq_pdb)

    # from Bio import pairwise2
    # from Bio.Seq import Seq
    # seq1 = Seq("ACCGGT")
    # seq2 = Seq("ACGT")
    # alignments = pairwise2.align.globalxx(seq1, seq2)



    aa=structure[0][chain_id]
    seq_list = []
    pdb_idx = []
    for ii in aa:
        # print(ii.resname)
        if ii.resname == 'HOH':
            break
        pdb_idx.append(ii.id[1]-1)
        aai = AA_shortener.get(ii.resname, '-')
        seq_list.append(aai)
    seq_pdb = ''.join(seq_list)
    seq_pdb_num = AA_convert(seq_pdb).squeeze().numpy()
    pdb_idx = np.asarray(pdb_idx)

    #First we need to make sure that we have a match in length between pdb_idx and seq_dssp
    if pdb_idx.shape[0] > seq_dssp_num.shape[0]:
        #Lets check whether we have seq_dssp in seq_pdb
        _, idx = match_arrays(seq_pdb_num, seq_dssp_num, exact=True)
        if idx < 0:
            raise ValueError("Didn't work")
        #We only keep the part of pdb_idx that matches seq_dssp
        pdb_idx = pdb_idx[idx:idx+seq_dssp_num.shape[0]]


    #We need to find the first index match between the pdb_idx and our sequence, in order to do this we take the largest chunk of the pdb_idx and compare it up against the sequence.
    pdb_idx_diff = pdb_idx[1:] - pdb_idx[:-1]
    cmp = np.where(pdb_idx_diff != 1)
    if cmp[0].shape[0] == 0:
        pdb_chunk = pdb_idx
    else:
        idx0 = 0
        max_chunk_size = -1
        for i in range(cmp[0].shape[0]):
            #Find largest chunk
            idx1 = cmp[0][i]+1
            chunk_size = idx1-idx0
            if chunk_size > max_chunk_size:
                max_chunk_size = chunk_size
                idx0_best = idx0
                idx1_best = idx1
            idx0 = idx1
        if len(pdb_idx)-idx0 > max_chunk_size:
            max_chunk_size = len(pdb_idx)-idx0
            idx0_best = idx0
            idx1_best = len(pdb_idx)
        pdb_chunk = pdb_idx[idx0_best:idx1_best]

    local_adj = - pdb_idx[0]
    try:
        score, idx = match_arrays(seq, seq_dssp_num[pdb_chunk+local_adj], exact=False)
        if score < 0.9*seq_dssp_num[pdb_chunk+local_adj].shape[0]:
            raise ValueError("Something went wrong")
        adjustment = idx - pdb_chunk[0]
        pdb_idx += adjustment
        dssp_out = 8 * np.ones_like(seq)
        if len(dssp_num) > len(seq):
            raise ValueError("Check it")
        else:
            dssp_out[pdb_idx] = dssp_num.squeeze()
            assert (seq_dssp_num == seq[pdb_idx]).all()
    except:
        #Last chance is if the seq_dssp_num is a subsequence of the seq itself
        score, idx = match_arrays(seq, seq_dssp_num, exact=False)
        if score < 0.9*seq_dssp_num.shape[0]:
            raise ValueError("Something went wrong")
        dssp_out = 8 * np.ones_like(seq)
        if len(dssp_num) > len(seq):
            raise ValueError("Check it")
        else:
            dssp_out[idx:idx+dssp_num.shape[0]] = dssp_num.squeeze()
    return dssp_out




def DSSP_wrapper2(file,DSSP_ALPHABET,seq,chain_number,chain_id):
    parser = MMCIFParser()
    structure = parser.get_structure("1MOT", file)
    models = structure[0]

    DSSP_results = DSSP_fnc(models, file, dssp='mkdssp')
    a_keys = list(DSSP_results.keys())

    unk_idx = -1
    models_cd = models.child_dict
    AA_convert = AA_converter(AA_alphabet, unk_idx)

    if chain_id == -1:
        idx, chain_id, seq_pdb = find_chain_fit(models_cd, seq, AA_shortener, AA_convert)


    chain_ids = [a_key[0] for a_key in a_keys]
    to_use = np.asarray([i == chain_id for i in chain_ids])

    DSSP_convert = AA_converter(DSSP_ALPHABET, -1)

    dssp = []
    seq_dssp = []
    for a_key,use_i in zip(a_keys,to_use):
        if use_i:
            tmp = DSSP_results[a_key]
            seq_dssp.append(tmp[1])
            dssp.append(tmp[2])
    dssp_num = DSSP_convert(dssp).squeeze().numpy()
    seq_dssp = ''.join(seq_dssp)
    seq_dssp_num = AA_convert(seq_dssp).squeeze().numpy()
    seq_l = AA_convert.reverse(seq)
    # seq_pdb_l = AA_convert.reverse(seq_pdb)

    seq1 = Seq(seq_l)
    seq2 = Seq(seq_dssp)
    alignments = pairwise2.align.globalxx(seq1, seq2)
    # for alignment in alignments:
    #     print(format_alignment(*alignment))
    # print("what now?")

    test_alignments = pairwise2.align.localds(seq1, seq2, blosum62, -10, -1)
    for alignment in test_alignments:
        print(format_alignment(*alignment))

    seqA = alignment.seqA
    seqB = alignment.seqB
    seqB_num = AA_convert(seqB).squeeze().numpy()
    seqA_num = AA_convert(seqA).squeeze().numpy()
    mB = seqB_num != 20
    mA = seqA_num != 20

    dssp_out = 8 * np.ones_like(seq)
    dssp_out[mB[mA]] = dssp_num[mA[mB]]
    return dssp_out




if __name__ == '__main__':
    pnet_folder = './pnet/'
    output_folder = './pnet_with_dssp/'
    pdb_folder = './pdbs/'

    # pnet_folder = '/media/tue/Data/Dropbox/ComputationalGenetics/data/pnet_with_msa_validation_new/'
    # output_folder = '/media/tue/Data/Dropbox/ComputationalGenetics/data/pnet_with_msa_and_dssp_validation/'
    # pdb_folder = './../../data/pdbs/'


    DSSP_ALPHABET = 'HBEGIPTS-'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(pdb_folder, exist_ok=True)
    pdbl = PDBList()

    search_command = pnet_folder + "*.npz"
    pnet_files = [f for f in sorted(glob.glob(search_command))]
    npnet_files = len(pnet_files)

    nrejected = 0
    print("Number of files remaining: {:}".format(npnet_files))
    t0 = time.time()
    for i,file_in in enumerate(pnet_files):

        t1 = time.time()
        d = np.load(file_in)
        seq = d['seq']
        id = str(d['id'])
        pdb_id = id.split('#')[-1].split('_')[0]
        # if pdb_id != '1S21':
        #     continue

        test = pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_folder, file_format='mmCif')

        t2 = time.time()
        try:
            chain_number = id.split('#')[-1].split('_')[1]
        except:
            chain_number = -1 #should be a number
        try:
            chain_id = id.split('#')[-1].split('_')[2]
        except:
            chain_id = -1 #Should be a letter


        file_pdb = "{:}{:}.cif".format(pdb_folder,pdb_id.lower())
        dssp_num = DSSP_wrapper2(file_pdb,DSSP_ALPHABET,seq,chain_number,chain_id)
        t3 = time.time()

        assert len(dssp_num) == len(seq)
        file_out = "{:}{:}.npz".format(output_folder, Path(file_in).stem)
        vars = d.files
        dd = {}
        for var in vars:
            tmp = d[var]
            dd[var] = tmp

        dd['dssp'] = dssp_num
        dd['DSSP_ALPHABET'] = DSSP_ALPHABET
        np.savez(file_out, **dd)
        print("{:}/{:}, accepted={:}, rejected={:}, Download_pdb={:2.2f}s, DSSP_wrapper={:2.2f}s, total={:2.2f}s".format(i+1,npnet_files,i+1-nrejected,nrejected,t2-t1,t3-t2,time.time()-t1))
