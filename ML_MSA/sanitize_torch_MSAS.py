import glob
import os
import time
import sys

import torch
from pathlib import Path


class CTError(Exception):
    def __init__(self, errors):
        self.errors = errors

try:
    O_BINARY = os.O_BINARY
except:
    O_BINARY = 0
READ_FLAGS = os.O_RDONLY | O_BINARY
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | O_BINARY
BUFFER_SIZE = 128*1024

def copyfile(src, dst):
    try:
        fin = os.open(src, READ_FLAGS)
        stat = os.fstat(fin)
        fout = os.open(dst, WRITE_FLAGS, stat.st_mode)
        for x in iter(lambda: os.read(fin, BUFFER_SIZE), b""):
            os.write(fout, x)
    finally:
        try: os.close(fin)
        except: pass
        try: os.close(fout)
        except: pass


if __name__ == '__main__':
    print(torch.__version__)

    folder_in = './'
    folder_out = './sanitized/'
    os.makedirs(folder_out,exist_ok=True)
    t0 = time.time()
    search_command = folder_in + "*.pt"
    files_in = [f for f in sorted(glob.glob(search_command))]

    min_msas_pr_seq = 100


    #First we load in all the information
    nfiles = len(files_in)
    seqs = []
    seqs_len = torch.empty(nfiles,dtype=torch.int64)
    n_msas = torch.empty(nfiles,dtype=torch.float32)
    nfaulty_files = 0
    with open("errors.log", "w+") as f:
        for i, file_in in enumerate(files_in):
            t1 = time.time()
            try:
                d = torch.load(file_in)
                seqs.append(d['seq'])
                seqs_len[i] = d['seq_len']
                n_msas[i] = d['n_msas_org']
            except:
                nfaulty_files += 1
                f.write("{:} \n".format(file_in))
                print("Errors found = {:}".format(nfaulty_files))
    if nfaulty_files > 0:
        print("Errors founds, exiting now")
        sys.exit()
    else:
        print("No errors found, continuing")
    t1 = time.time()

    #Now we sanitize it
    #Make sure that only data with sufficient MSAS is saved.
    idx_to_keep = n_msas >= min_msas_pr_seq
    n_insufficient_msas = torch.sum(~ idx_to_keep)

    files_in = [var for var,con in zip(files_in,idx_to_keep) if con]
    seqs = [var for var,con in zip(seqs,idx_to_keep) if con]
    seqs_len = seqs_len[idx_to_keep]
    n_msas = n_msas[idx_to_keep]

    t2 = time.time()
    #Make sure that duplicates gets removed.

    sort_idx = torch.argsort(seqs_len)
    files_in = [files_in[i] for i in sort_idx]
    seqs = [seqs[i] for i in sort_idx]
    seqs_len = seqs_len[sort_idx]
    n_msas = n_msas[sort_idx]
    nfiles = len(files_in)

    seqs_len_unique,counts =torch.unique_consecutive(seqs_len, return_counts=True)
    idx0=0
    idx_to_keep = torch.ones(nfiles,dtype=torch.bool)
    n_duplicates = 0
    for i in counts:
        idx1 = idx0+i
        local_idx_to_keep = torch.ones(i,dtype=torch.bool)
        seqs_i = torch.stack(seqs[idx0:idx1])
        seqs_uni, counts_i = torch.unique(seqs_i,dim=0, return_counts=True)
        duplicate_idx = counts_i > 1
        n_duplicates += torch.sum(counts_i - 1)
        duplicate_seq = seqs_uni[duplicate_idx,:]
        for j in range(duplicate_seq.shape[0]):
            dseq = duplicate_seq[j,:]
            seq_idx = torch.nonzero((dseq == seqs_i).all(dim=1)).squeeze()
            local_idx_to_keep[seq_idx] = False
            n_msas_i = n_msas[idx0+seq_idx]
            jj = torch.argmax(n_msas_i)
            local_idx_to_keep[seq_idx[jj]] = True
            idx_to_keep[idx0:idx1] = local_idx_to_keep
            print("duplicate found, n_msas = {:}".format(n_msas_i))
        idx0=idx1

    files_in = [var for var,con in zip(files_in,idx_to_keep) if con]
    seqs = [var for var,con in zip(seqs,idx_to_keep) if con]
    seqs_len = seqs_len[idx_to_keep]
    n_msas = n_msas[idx_to_keep]
    t3 = time.time()

    print("Sanitizing complete, saving files...")
    for i, file_in in enumerate(files_in):
        file_out = "{:}{:}.pt".format(folder_out, Path(file_in).stem)
        copyfile(file_in, file_out)
    t4 = time.time()
    print("n duplicates found={:}, n_insufficient_msas found={:}".format(n_duplicates, n_insufficient_msas))
    print("Done, t_load={:2.2f},t_min_nmsas={:2.2f},t_duplicates={:2.2f},t_copy={:2.2f},total={:2.2f}".format(t1-t0,t2-t1,t3-t2,t4-t3,t4-t0))

