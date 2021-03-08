import glob
import os
import time
from pathlib import Path
import string
from xopen import xopen

import torch

from ML_MSA.MSA_reader import read_a2m_gz_file
from supervised.dataloader_utils import AA_LIST
import multiprocessing as mp


class AA_converter(object):
    def __init__(self, aa_list, unk_idx):
        self.aa_list = aa_list
        self.tok_to_idx = {tok: i for i, tok in enumerate(self.aa_list)}
        self.unk_idx = unk_idx

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



def read_a2m_gz_file(a2mfile, AA_LIST, unk_idx, max_seq_len=9999999,min_seq_len=-1, verbose=False):
    """
    This will read and return the MSAs from a single a2m.gz file.
    The MSA will be returned as a numpy array.
    The MSA will have the shape (n x l) where n is the number of sequence alignments found, and l is the sequence length.
    The MSA will be returned as numbers ranging from 0-20, which cover the 20 common amino acids as well as '-' which contains everything else.
    The a2m.gz format is expected to be similar to the following example:
    >XXXX_UPI0000E497C4/159-301 [subseq from] XXXX_UPI0000E497C4
    --DERQKTLVENTWKTLEKNTELYGSIMFAKLTTDHPDIGKLFPFGgkNLTYgellVDPD
    VRVHGKRVIETLGSVVEDLDDmelVIQILEDLGQRHNA-YNAKKTHIIAVGGALLFTIEE
    ALGAGFTPEVKAAWAAVYNIVSDTMS----
    >XXXX_UPI0000E497C4/311-417 [subseq from] XXXX_UPI0000E497C4
    ---AREQELVQKTWGVLSLDTEQHGAAMFAKLISAHPAVAQMFPFGeNLSYsqlvQNPTL
    RAHGKRVMETIGQTVGSLDDldiLVPILRDLARRHVG-YSVTRQHFEGPKE---------
    -----------------------------
    >1A00_1_A
    VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGK
    KVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPA
    VHASLDKFLASVSTVLTSKYR
    Where each line starting with ">" signals a header line, meaning that a new sequence is comming on the following lines.
    Note that the last sequence in the sequence should be the origin sequence.
    So in the above example we have 2 sequences and the origin sequence.
    Note this has only been tested on windows.
    """
    t0 = time.time()
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    seqs = []
    seq = ""
    check_length = True

    AA = AA_converter(AA_LIST,unk_idx)

    with xopen(a2mfile) as fin:
        for text in fin:
            # skip labels
            if text[0] == '>':
                if seq != "":
                    # remove lowercase letters and right whitespaces
                    seqs.append(seq)
                    seq = ""
                    if check_length:
                        if len(seqs[-1]) > max_seq_len or len(seqs[-1]) <= min_seq_len:
                            if verbose:
                                print("Skipping MSA since sequence length is outside allowed range: {:}. Time taken {:2.2f}".format(len(seqs[-1]),time.time()-t0))
                            msa = None
                            return msa
            else:
                seq += text.rstrip().translate(table)
    seqs.append(seq)  # We include the parent protein in the MSA sequence
    msa = AA(seqs)
    return msa




def read_and_save_a2m_file(file):
    AA_LIST = 'ACDEFGHIKLMNPQRSTVWY-'
    max_seq_len = 600
    min_seq_len = 1
    max_samples = 30000
    unk_idx = 20
    folder_in = 'D:/raw_MSA/'
    folder_out = 'D:/MSAS/'
    extension = '.a2m.gz'
    # folder_in = '/home/tue/Dropbox/ComputationalGenetics/data/raw_MSA_small_sample/'
    # folder_out = '/home/tue/Dropbox/ComputationalGenetics/data/raw_MSA_small_sample/out/'
    file_in = "{:}{:}{:}".format(folder_in,file,extension)
    file_out = "{:}{:}.pt".format(folder_out,file)

    msas = read_a2m_gz_file(file_in, AA_LIST=AA_LIST, unk_idx=unk_idx, max_seq_len=max_seq_len, min_seq_len=min_seq_len,verbose=True)
    if msas is None:
        return
    else:
        seq = msas[-1, :]
        msas = msas[:-1, :]
        n_msa = msas.shape[0]
        seq_len = msas.shape[1]

        if n_msa > max_samples:
            indices = torch.randperm(n_msa)
            msas_s = msas[indices[:max_samples], :]
        else:
            msas_s = msas

        data = {'seq': seq.to(dtype=torch.int8),
                'msas': msas_s.to(dtype=torch.int8),
                'seq_len': seq_len,
                'n_msas_org': n_msa
                }
        torch.save(data, file_out)


if __name__ == '__main__':
    folder_in = 'D:/raw_MSA/'
    folder_out = 'D:/MSAS/'
    os.makedirs(folder_out,exist_ok=True)
    search_command = folder_in + "*.a2m.gz"
    a2mfiles = [f for f in sorted(glob.glob(search_command))]
    # max_seq_len = 600
    # min_seq_len = 1
    # max_samples = 30000
    # unk_idx = 20
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

    # base_infiles = Path(Path(a2mfiles).stem).stem
    base_infiles = [Path(Path(x).stem).stem for x in a2mfiles]

    # base_outfiles = Path(outfiles).stem
    base_outfiles = [Path(x).stem for x in outfiles]

    basefiles_in = [x for x in base_infiles if x not in base_outfiles]


    nfiles = len(basefiles_in)
    print("Remaining files= {:}".format(nfiles))
    t0 = time.time()


    with mp.Pool(processes=2) as pool:
        for i in pool.imap_unordered(read_and_save_a2m_file, basefiles_in):
            pass
        pool.close()
        pool.join()

    t1 = time.time()
    print("Time taken {:2.2f}, Average time pr file {:2.2f}s".format(t1-t0,(t1-t0)/nfiles))
