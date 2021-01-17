import glob
import os

import torch.nn.functional as F

import torch
import esm
import time
import numpy as np

from ML_MSA.MSA_generators import mask_scheme1, MSA_once, mask_scheme_cov, DummyModel, \
    setup_reference_matrix_job, compute_reference_matrix, setup_conditional_prob_matrix_job, \
    compute_conditional_prob_matrix, compute_kl_divergence_matrix, ModelWrapper
from ML_MSA.MSA_reader import read_a2m_gz_file
from ML_MSA.MSA_utils import msa2weight, msa2pssm, msa2cov, setup_protein_comparison, AA_converter, Translator
from supervised.dataloader_pnet import parse_pnet
from supervised.network_transformer import tr2DistSmall
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') #TkAgg

def compute_kl_div_matrix(seqidx,mask_idx,aa_list,maxresidues):
    device = seqidx.device
    seqs_ref, rowidx,colidx = setup_reference_matrix_job(seqidx,mask_id=mask_idx) # This may as well be returned as a matrix I think, and then it can be naturally saved in that afterwards as well.
    ref_matrix = compute_reference_matrix(model, seqs_ref, rowidx, colidx, maxresidues=maxresidues, device=device, ncategories=len(aa_list))

    seqs_con, maskids,targetids,tokenids = setup_conditional_prob_matrix_job(seqidx,alphabet=aa_list,mask_id=mask_idx)
    con_matrix = compute_conditional_prob_matrix(model, seqs_con, maskids, targetids, tokenids, maxresidues=maxresidues, device=device, ncategories=len(aa_list))

    kl_matrix = compute_kl_divergence_matrix(ref_matrix,con_matrix)
    return kl_matrix,con_matrix,ref_matrix




if __name__ == "__main__":
    filename = 'dummy'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    outputfolder = './../results/ML_MSA/'
    os.makedirs(outputfolder,exist_ok=True)
    model_facebook, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    translator = Translator()
    model = ModelWrapper(model_facebook,translator)
    # model = DummyModel(aa_list,unknown_idx = mask_idx)

    pnet_alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    mask_idx = 20
    AA_to_idx = AA_converter(pnet_alphabet,mask_idx)
    maxresidues = 1000

    seq = ['ACDEFGHIKLMNPQRSTVWY-']
    seqidx = AA_to_idx(seq)
    seqidx = seqidx.squeeze().to(device)

    kl_matrix,con_matrix,ref_matrix = compute_kl_div_matrix(seqidx, mask_idx, pnet_alphabet, maxresidues)

    dat = {"masked_pairs": ref_matrix,
           "conditional_prob": con_matrix,
           "kl_div": kl_matrix
           }
    save = "{:}{}.pt".format(outputfolder,filename)

    torch.save(dat, save)

    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 10))
    plt.imshow(kl_matrix.cpu())
    plt.colorbar()
    plt.savefig("{:}{:}_kldiv.png".format(outputfolder,filename))
