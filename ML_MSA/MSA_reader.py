import glob
import time
import torch
import numpy as np
from xopen import xopen
import numpy as np
import string

from ML_MSA.MSA_utils import AA_converter
from supervised.dataloader_utils import AA_LIST


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





if __name__ == "__main__":
    MSA_folder = "./../data/MSA/"
    search_command = MSA_folder + "*.a2m.gz"
    a2mfiles = [f for f in sorted(glob.glob(search_command))]
    max_seq_len = 320
    min_seq_len = 80
    max_samples = 30000
    nsub_size = 10000
    n_repeats = 9
    unk_idx = 20
    for i, a2mfile in enumerate(a2mfiles):
        t0 = time.time()
        msas = read_a2m_gz_file(a2mfile, AA_LIST=AA_LIST, unk_idx=unk_idx, max_seq_len=max_seq_len, min_seq_len=min_seq_len, verbose=True)
        if msas is None:
            continue
        else:
            n = msas.shape[0]
            l = msas.shape[1]
            print("# MSAs = {:}, sequence length {:}, time taken: {:2.2f}".format(n, l, time.time()- t0))