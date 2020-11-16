import os

import numpy as np
from src.dataloader_pnet import parse_pnet

if __name__ == '__main__':
    pnetfile = './../data/training_30.pnet'
    output_folder = './../data/train_npz/'
    # pnetfile = './../data/testing.pnet'
    # output_folder = './../data/test_npz/'

    os.makedirs(output_folder, exist_ok=True)
    args = parse_pnet(pnetfile, min_seq_len=-1, max_seq_len=1000, use_entropy=True, use_pssm=True, use_dssp=False, use_mask=False, use_coord=True)

    ids = args['id']
    r1 = args['r1']
    r2 = args['r2']
    r3 = args['r3']
    pssm = args['pssm']
    entropy = args['entropy']
    seq = args['seq']
    for i,id in enumerate(ids):
        filename = "{:}{:}.npz".format(output_folder,id)
        entropy_i = entropy[i]
        np.savez(file=filename, seq=seq[i],pssm=pssm[i],entropy=entropy_i[None,:],r1=r1[i].T,r2=r2[i].T,r3=r3[i].T,id=id)

