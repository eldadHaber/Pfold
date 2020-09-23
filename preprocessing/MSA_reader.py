import time
import gzip
import numpy as np
import string
import glob

from srcOld.dataloader_pnet import read_record, parse_pnet

AA_LIST = 'ACDEFGHIKLMNPQRSTVWY-'


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

        alphabet = np.array(list(AA_LIST), dtype='|S1').view(np.uint8)

        msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
        for i in range(alphabet.shape[0]):
            msa[msa == alphabet[i]] = i

        # treat all unknown characters as gaps
        msa[msa > 20] = 20
        msas.append(msa)
    return msas, proteins



def read_a2m_gz_file(a2mfile,AA_LIST=AA_LIST):
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

    encoding = 'utf-8'
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
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
    protein =  seq # A2m ends with the parent protein.
    seqs.append(seq) #We include the parent protein in the MSA sequence
    # convert letters into numbers

    alphabet = np.array(list(AA_LIST), dtype='|S1').view(np.uint8)

    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20
    return msa, protein



def weight_msa(msa_1hot, cutoff):
    """
    Finds the weight for each MSA given an identity cutoff. Typical values for the cutoff are 0.8.
    NOTE that this will take a long time for MSAs with a lot of sequences and might be possible to speed up by using generators or something similar.
    The reason why this takes such a long time is that tensordot returns a (n x n) matrix where n is the number of sequences in the MSA.
    Follows the procedures used by trRossetta
    """
    id_min = msa_1hot.shape[1] * cutoff
    id_mtx = np.tensordot(msa_1hot, msa_1hot, axes=([1, 2], [1, 2]))
    id_mask = id_mtx > id_min
    w = 1.0 / np.sum(id_mask, axis=-1)
    return w


def msa2pssm(msa_1hot, w):
    """
    Computes the position scoring matrix, f_i, given an MSA and its weight.
    Furthermore computes the sequence entropy, h_i.
    Follows the procedures used by trRossetta
    """
    neff = np.sum(w)
    f_i = np.sum(w[:, None, None] * msa_1hot, axis=0) / neff
    h_i = np.sum(- (f_i + 1e-9) * np.log(f_i + 1e-9), axis=1)
    return np.concatenate([f_i, h_i[:, None]], axis=1)

def dca(msa_1hot, w, penalty=4.5):
    """
    This follows the procedures used by trRossetta.
    Computes the covariance and inverse covariance matrix (equation 2), as well as the APC (equation 4).
    """

    nr, nc, ns = msa_1hot.shape
    x = msa_1hot.reshape(nr, nc * ns)
    num_points = np.sum(w) - np.sqrt(np.mean(w))
    mean = np.sum(x * w[:, None], axis=0, keepdims=True) / num_points
    x = (x - mean) * np.sqrt(w[:, None])
    cov = np.matmul(x.T, x) / num_points

    cov_reg = cov + np.eye(nc * ns) * penalty / np.sqrt(np.sum(w))
    inv_cov = np.linalg.inv(cov_reg)
    x1 = inv_cov.reshape(nc, ns, nc, ns)
    x2 = x1.transpose((0,2,1,3))
    features = x2.reshape(nc, nc, ns * ns)

    x3 = np.sqrt(np.sum(np.square(x1[:, :-1, :, :-1]), axis=(1,3))) * (1 - np.eye(nc))
    apc = np.sum(x3, axis=0, keepdims=True) * np.sum(x3, axis=1, keepdims=True) / np.sum(x3)
    contacts = (x3 - apc) * (1 - np.eye(nc))
    return np.concatenate([features, contacts[:, :, None]], axis=2)

if __name__ == "__main__":
    # path = "F:/Globus/raw_subset/" # Path to a folder with a2m.gz files in it.
    # msas, proteins = read_a2m_gz_folder(path)
    #
    # msa = msas[0]
    #
    # t0 = time.time()
    # msa_1hot = np.eye(21, dtype=np.float32)[msa]
    # msa_1hot_r = msa_1hot[:,:,:-1]
    # cutoff = 0.9999
    # w = weight_msa(msa_1hot,cutoff)
    # t1 = time.time()
    # pssm = msa2pssm(msa_1hot, w)
    # # pssm2 = pssm[:,:-1]
    # # pssm3 = pssm2[:,:-1]
    # # tmp = np.sum(pssm3,axis=1)
    # # pssm3_norm = pssm3/tmp[:,None]
    # # p = pssm3_norm.T
    # # p[:,0:4]
    # t2 = time.time()
    # f2d_dca = dca(msa_1hot, w)
    # t3 = time.time()
    # print("time taken: {:2.2f}, {:2.2f}, {:2.2f}".format(t1-t0,t2-t1,t3-t2))


    # This second part is meant to be the real deal
    # Here we load the full protein net training set,
    # and then go through the MSA set iteratively, and compare them against the protein net set.
    pnet_file = "D:/pytorch/Pfold/data/training_wrong.pnet"
    seqs, r1,r2,r3, seqs_len = parse_pnet(pnet_file, max_seq_len=-1)

    seqs_len_unique, counts = np.unique(seqs_len, return_counts=True)
    a=np.digitize(seqs_len,bins=seqs_len_unique,right=True)
    lookup = {}  # create an empty dictionary
    # Make a lookup table such that t[seq_len] = idx
    for i, seq_len_unique in enumerate(seqs_len_unique):
        lookup[seq_len_unique] = i

    # Next we create the list of arrays, and then afterwards we populate them
    seqs_list = []
    seqs_list_org_id = []
    for seq_len_unique,count in zip(seqs_len_unique,counts):
        tmp = np.empty((count,seq_len_unique), dtype=np.int32)
        seqs_list.append(tmp)
        tmp = np.empty((count,1), dtype=np.int32)
        seqs_list_org_id.append(tmp)

    counter = np.zeros_like(counts)
    for i,(seq,seq_len) in enumerate(zip(seqs,seqs_len)):
        seq_idx = lookup[seq_len]
        counter_idx = counter[seq_idx]
        seqs_list[seq_idx][counter_idx,:] = seq
        seqs_list_org_id[seq_idx][counter_idx] = i
        counter[seq_idx] += 1


    n = len(seqs)
    matches = np.zeros(n, dtype=np.int32)

    # I think the correct way to do this is to sort pnet, by length of the proteins, and group all proteins with the same length in an nd.array
    # Make a lookup table with idx -> seq_len
    # Then the comparison is only against all elements in that particular group.

    n_matches = 0
    MSA_not_in_protein_net = 0
    MSA_in_protein_net_more_than_once = 0

    MSA_folder = "F:/Globus/raw_subset/"
    # MSA_folder = "F:/Globus/raw_subset/"
    search_command = MSA_folder + "*.a2m.gz"
    # a2mfiles = [f for f in glob.glob(search_command)]
    # for a2mfile in a2mfiles:
        # msa, protein = read_a2m_gz_file(a2mfile)
    protein = seqs[-4]
    seq_len = len(protein)

    seq_idx = lookup[seq_len]
    seqs_i = seqs_list[seq_idx]
    res = np.mean(protein == seqs_i,axis=1) == 1
    if np.sum(res) == 1:
        n_matches += 1
        org_id = seqs_list_org_id[seq_idx][res]
        matches[org_id] += 1
    elif np.sum(res) == 0:
        MSA_not_in_protein_net += 1
    else:
        MSA_in_protein_net_more_than_once += 1
print("done")