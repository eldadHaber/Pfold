import glob
import os

import numpy as np
from pathlib import Path

#This routine assumes you have already converted MSAS into torch format, using process_raw_MSAS. Not that the MSAS should already have been sanitized (Check for duplicates and remove any MSAS wit insufficient information ect.)
#Furthermore it assumes you have proteinnet data converted into torch format as well, which can be done using dataloader_pnet

import torch

if __name__ == '__main__':
    pnet_folder = './../../../data/msa_test_env/pnet/'
    msa_folder = './../../../data/msa_test_env/msa/'
    output_folder = './../../../data/msa_test_env/out/'
    os.makedirs(output_folder, exist_ok=True)
    #This code will read in all the data in pnet and store it in memory.
    #It will then go through the data in the MSA folder one by one and pair try to pair it up against pnet data, when a match is found the data will be save to output combined, and the data will be removed from pnet, such that the search space gets smaller and smaller as it progresses


    #First we load pnet into memory. We need the sequence, and filename, though we can probably store all information in memory so we do that.
    search_command = pnet_folder + "*.npz"
    pnet_files = [f for f in sorted(glob.glob(search_command))]
    npnet_files = len(pnet_files)

    seqs = []
    seqs_len = torch.empty(npnet_files,dtype=torch.int64)

    for i,file in enumerate(pnet_files):
        d = np.load(file)
        seq = torch.from_numpy(d['seq'])
        seqs.append(seq)
        seqs_len[i] = seq.shape[0]

    idx_sort = torch.argsort(seqs_len)
    seqs_len = seqs_len[idx_sort]
    seqs = [seqs[i] for i in idx_sort]
    pnet_files = [pnet_files[i] for i in idx_sort]

    seqs_len_unique,idx_seqs_len, count_seqs_len = torch.unique_consecutive(seqs_len,return_inverse=True,return_counts=True)
    idx0 = 0
    seqs_stacked_by_len = []
    for i in range(seqs_len_unique.shape[0]):
        idx1 = idx0 + count_seqs_len[i]
        seqs_stacked_by_len.append(torch.stack(seqs[idx0:idx1]))
        idx0 = idx1


    #assume we have a sequence
    search_command = msa_folder + "*.pt"
    msa_files = [f for f in sorted(glob.glob(search_command))]
    nmsa_files = len(msa_folder)
    for i,file in enumerate(msa_files):
        d = torch.load(file)
        seq = d['seq']
        seq_len = d['seq_len']
        len_idx = torch.nonzero(seq_len == seqs_len_unique)
        if len_idx.shape[0] > 0: #The seq_len exist in proteinnet, do we have any sequence that match?
            seq_idx = torch.nonzero((seq == seqs_stacked_by_len[len_idx]).all(dim=1)).squeeze()
            if seq_idx.shape[0] > 0:
                # We found a match!, we save it immediately as numpy
                file_out = "{:}{:}.npz".format(output_folder, Path(file).stem)
                d_pnet = np.load(pnet_files[seq_idx.numpy()])
                np.savez(file=file_out, seq=seq.numpy(), pssm=d_pnet['pssm'], entropy=d_pnet['entropy'], rCa=d_pnet['rCa'],
                         rCb=d_pnet['rCb'], rN=d_pnet['rN'], id=d_pnet['id'], log_units=d_pnet['log_units'], AA_LIST=d_pnet['AA_LIST'],
                         msa=d['msas'].numpy(), nmsa_org=d['n_msas_org'])
                #Finally we can remove the sequence from seqs_stacked_by_len if we think it will be faster in the long run, which it probably will be.
