import glob
import os
import time
from pathlib import Path

import torch

from ML_MSA.MSA_reader import read_a2m_gz_file
from supervised.dataloader_utils import AA_LIST


# folder_in = 'F:/Globus/raw/'
folder_in = 'F:/MSAS_old/'
# folder_in = 'F:/Globus/raw_subset/'
folder_out = 'F:/MSAS/'
os.makedirs(folder_out,exist_ok=True)
search_command = folder_in + "*.pt"
a2mfiles = [f for f in sorted(glob.glob(search_command))]
max_seq_len = 600
min_seq_len = 1
max_samples = 30000
unk_idx = 20
#
search_command = folder_out + "*.pt"
outfiles = [f for f in sorted(glob.glob(search_command))]
# for i, file in enumerate(torchfiles):
#     t0 = time.time()
#     data = torch.load(file)
#     msa = data['msas']
#     seq = data['seq']
#     t1 = time.time()
#     print("Time taken {:2.2f}".format(t1-t0))
#

nfiles = len(a2mfiles)
t0 = time.time()
for i, a2mfile in enumerate(a2mfiles):
    t1 = time.time()
    # msas = read_a2m_gz_file(a2mfile, AA_LIST=AA_LIST, unk_idx=unk_idx, max_seq_len=max_seq_len, min_seq_len=min_seq_len,
    #                         verbose=True)
    d = torch.load(a2mfile)
    t2 = time.time()

    data = {'seq':d['seq'].to(dtype=torch.int8),
            'msas': d['msas'].to(dtype=torch.int8),
            'seq_len': d['seq_len'],
            'n_msas_org': d['n_msas_org']
            }
    t3 = time.time()

    filename = Path(Path(a2mfile).stem).stem

    torch.save(data, "{:}{:}.pt".format(folder_out,filename))
    tn = time.time()
    print("{:}/{:} , Seq_len {:},time taken: {:2.2f}s,read{:2.2f},convert{:2.2f},save{:2.2f},ETA: {:2.2f}h".format(i+1,nfiles,data['seq_len'],tn- t1,t2-t1,t3-t2,tn-t3,(nfiles-(i+1))*(time.time()-t0)/(i+1)/3600))
