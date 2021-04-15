import glob

import numpy as np
import torch

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
        l = seq.shape[-1]
        seq_letters = []
        for i in range(l):
            seq_letters.append(self.idx_to_tok.get(seq_num[i],self.unk_tok))
        seq_l = ''.join(seq_letters)
        return seq_l



folder_to_check = '/media/tue/Data/Dropbox/ComputationalGenetics/data/pnet_with_msa_and_dssp_validation/'

search_command = folder_to_check + "*.npz"
pnet_files = [f for f in sorted(glob.glob(search_command))]
npnet_files = len(pnet_files)

AA_convert = AA_converter(AA_alphabet, -1)

print("Checking {:} files".format(npnet_files))
for file in pnet_files:
    d = np.load(file)
    rCa = d['rCa']
    dssp = d['dssp']
    m = rCa[0,:] == 0
    seq = d['seq']
    seq_alpha = AA_convert.reverse(seq)
    dssp_sel = dssp[m]
    assert (dssp_sel == 8).all()

print("Everything seems good")


