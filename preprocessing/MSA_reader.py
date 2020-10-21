import time
import gzip
from xopen import xopen
import numpy as np
import string
import glob

from srcOld.dataloader_utils import AA_LIST

def read_a2m_gz_folder(folder,AA_LIST=AA_LIST):
    """
    This will read and return the MSAs from all a2m.gz files in the folder given.
    The MSAs will be returned as a list of numpy arrays, where each list element/numpy array corresponds to a a2m.gz file.
    Each MSA will have the shape (n x l) where n is the number of sequence alignments found, and l is the sequence length.
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

    search_command = folder + "*.a2m.gz"
    a2mfiles = [f for f in glob.glob(search_command)]
    encoding = 'utf-8'
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    proteins = []
    msas = []
    for a2mfile in a2mfiles:
        seqs = []
        seq = ""
        # read file line by line
        with gzip.open(a2mfile,'r') as fin:
            for line in fin:
                text = line.decode(encoding)
                # skip labels
                if text[0] == '>':
                    if seq != "":
                        # remove lowercase letters and right whitespaces
                        seqs.append(seq)
                        seq = ""
                else:
                    seq += text.rstrip().translate(table)
        proteins.append(seq) # A2m ends with the parent protein.
        seqs.append(seq) #We include the parent protein in the MSA sequence
        # convert letters into numbers

        alphabet = np.array(AA_LIST, dtype='|S1').view(np.uint8)

        msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
        for i in range(alphabet.shape[0]):
            msa[msa == alphabet[i]] = i

        # treat all unknown characters as gaps
        msa[msa > 20] = 20
        msas.append(msa)
    return msas, proteins

def read_a2m_gz_file(a2mfile, AA_LIST=AA_LIST, max_seq_len=-1,min_seq_len=9999999):
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

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    seqs = []
    seq = ""
    # read file line by line
    # with gzip.open(a2mfile,'r') as fin:
    t0 = time.time()
    check_length = True
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
                            msa = None
                            return msa
            else:
                seq += text.rstrip().translate(table)
    seqs.append(seq)  # We include the parent protein in the MSA sequence
    # convert letters into numbers
    t1 = time.time()
    alphabet = np.array(AA_LIST, dtype='|S1').view(np.uint8)
    # print("msas: {:}".format(len(seqs)))
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)

    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20
    t2 = time.time()
    # print("read {:2.2f}, msa {:2.2f}".format(t1-t0, t2-t1))
    return msa




if __name__ == "__main__":
    MSA_folder = "F:/Globus/raw_subset/"
    search_command = MSA_folder + "*.a2m.gz"
    a2mfiles = [f for f in glob.glob(search_command)]
    max_seq_len = 320
    min_seq_len = 80
    max_samples = 30000
    nsub_size = 10000
    n_repeats = 9
    for i, a2mfile in enumerate(a2mfiles):
        t0 = time.time()
        msas = read_a2m_gz_file(a2mfile, max_seq_len=max_seq_len, min_seq_len=min_seq_len)
        if msas is None:
            continue
        else:
            n = msas.shape[0]
            l = msas.shape[1]
            print("# MSAs = {:}, time taken: {:2.2f}".format(n, time.time()- t0))

