import glob
import torch.nn.functional as F

import torch
import esm
import time
import numpy as np

from ML_MSA.MSA_generators import mask_scheme1, MSA_once, mask_scheme_cov, mask_scheme_cov_ref, mask_scheme_pssm
from ML_MSA.MSA_reader import read_a2m_gz_file
from ML_MSA.MSA_utils import msa2weight, msa2pssm, msa2cov, setup_protein_comparison, AA_converter
from src.dataloader_pnet import parse_pnet
from src.network_transformer import tr2DistSmall
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') #TkAgg

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
# model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
aa_list = alphabet.all_toks
aa_eff = aa_list[4:-4]
batch_converter = alphabet.get_batch_converter()

AA = AA_converter(aa_list,alphabet.unk_idx)
test_toks = AA(aa_eff)
n_test_toks = len(test_toks)
# batch_converter = BatchConverter(alphabet)

# seqs = ["MYMKK", "MYYKK"]
# strs, tokens = batch_converter(seqs)

# data = [("protein1", "MYLYQKIKN"), ("protein2", "MNAKYD")]
# batch_labels, batch_strs, batch_tokens = batch_converter(data)

#Load sequence
folder = './../data/casp11_validation/'
search_command = folder + "*.npz"
npzfiles = [f for f in glob.glob(search_command)]

for ii,npzfile in enumerate(npzfiles):
    dat = np.load(npzfile)
    seq = torch.from_numpy(dat['seq']).to(dtype=torch.int64,device=device)
    l = seq.shape[0]

    seqs,masks = mask_scheme_pssm(seq, mask_id=32, prepend_idx=0, append_idx=2)
    nb = 10000//l
    model.to(device)
    n = seqs.shape[0]
    maxiter = np.int(np.ceil(n/nb))
    pssm25 = torch.empty((n,n_test_toks),dtype=torch.float32,device=device)
    for i in range(maxiter):
        i0 = i*nb
        i1 = (i+1)*nb
        seq_batch = seqs[i0:i1,:]
        mask_batch = masks[i0:i1,:]
        with torch.no_grad():
            results = model(seq_batch)
            pred = results['logits']
            pred = pred[mask_batch,:]
            pred = pred[:,4:-4]
            prob = torch.softmax(pred, dim=1)
            idx = torch.argmax(mask_batch.float(),dim=1)
            pssm25[idx-1, :] = prob
    # pssm20 = torch.empty((n,20),dtype=torch.float32,device=device)
    # pssm20[:,0] = pssm25[:,1]
    # pssm20[:, 1] = pssm25[:, 19]
    # pssm20[:, 2] = pssm25[:, 9]
    # pssm20[:, 3] = pssm25[:, 5]
    # pssm20[:, 4] = pssm25[:, 14]
    # pssm20[:, 5] = pssm25[:, 2]
    # pssm20[:, 6] = pssm25[:, 17]
    # pssm20[:, 7] = pssm25[:, 8]
    # pssm20[:, 8] = pssm25[:, 11]
    # pssm20[:, 9] = pssm25[:, 0]
    # pssm20[:, 10] = pssm25[:, 16]
    # pssm20[:, 11] = pssm25[:, 13]
    # AA_DICT = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7', 'K': '8', 'L': '9',
    #            'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18',
    #            'Y': '19', '-': '20'}
    #
    # pssm20[:, 12] = pssm25[:, 10]
    # pssm20[:, 13] = pssm25[:, 12]
    # pssm20[:, 14] = pssm25[:, 6]
    # pssm20[:, 15] = pssm25[:, 4]
    # pssm20[:, 16] = pssm25[:, 7]
    # pssm20[:, 17] = pssm25[:, 3]
    # pssm20[:, 18] = pssm25[:, 18]
    # pssm20[:, 19] = pssm25[:, 15]
    #
    # AA_DICT = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7', 'K': '8', 'L': '9',
    #            'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18',
    #            'Y': '19', '-': '20'}
    pssm = pssm25.cpu().numpy().T
    id = dat['id']
    save = "./../data/casp11_validation_ml/{:}.npz".format(id)
    np.savez(save, seq=dat['seq'], pssm=pssm, entropy=dat['entropy'],r1=dat['r1'],r2=dat['r2'],r3=dat['r3'],id=dat['id'])
input("Press Enter to continue...")
print("done")