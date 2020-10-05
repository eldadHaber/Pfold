import time
import gzip
from xopen import xopen
import numpy as np
import string
import glob

from preprocessing.ANN import ANN_hnsw, ANN_sparse
from srcOld.dataloader_utils import AA_DICT, DSSP_DICT, NUM_DIMENSIONS, MASK_DICT, SeqFlip, ListToNumpy, \
    DrawFromProbabilityMatrix
import re
from itertools import compress



from srcOld.dataloader_pnet import read_record, parse_pnet, separate_coords

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



def read_a2m_gz_file_old(a2mfile,AA_LIST=AA_LIST):
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
    seqs.append(seq) #We include the parent protein in the MSA sequence
    # convert letters into numbers

    alphabet = np.array(list(AA_LIST), dtype='|S1').view(np.uint8)

    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20
    return msa




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
    alphabet = np.array(list(AA_LIST), dtype='|S1').view(np.uint8)
    # print("msas: {:}".format(len(seqs)))
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)

    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20
    t2 = time.time()
    # print("read {:2.2f}, msa {:2.2f}".format(t1-t0, t2-t1))
    return msa


def weight_msa(msa_1hot, cutoff):
    """
    Finds the weight for each MSA given an identity cutoff. Typical values for the cutoff are 0.8.
    NOTE that this will take a long time for MSAs with a lot of sequences and might be possible to speed up by using generators or something similar.
    The reason why this takes such a long time is that tensordot returns a (n x n) matrix where n is the number of sequences in the MSA.
    Follows the procedures used by trRossetta
    """
    assert cutoff < 1
    id_min = msa_1hot.shape[1] * cutoff
    id_mtx = np.tensordot(msa_1hot, msa_1hot, axes=([1, 2], [1, 2]))
    id_mask = id_mtx > id_min
    w = 1.0 / np.sum(id_mask, axis=-1)
    return w




def weight_msa_fast1(msa_1hot):
    """
    Finds the weight for each MSA given an identity cutoff. Typical values for the cutoff are 0.8.
    """
    import sparse as sp
    msa_1hot_sp = sp.COO(msa_1hot)
    id_mtx = sp.tensordot(msa_1hot_sp, sp.tensordot(msa_1hot_sp,np.ones(msa_1hot.shape[0]),axes=([0],[0])), axes=([1, 2], [0, 1]))
    w = 1.0 / id_mtx
    return w

def weight_msa_fast2(msa_1hot):
    """
    Finds the weight for each MSA given an identity cutoff. Typical values for the cutoff are 0.8.
    """
    msa_1hot = msa_1hot.reshape(msa_1hot.shape[0],-1)
    id_mtx = msa_1hot @ (msa_1hot.T @ np.ones(msa_1hot.shape[0]))
    w = 1.0 / id_mtx
    return w



def weight_msa_fast3(msa_1hot):
    """
    Finds the weight for each MSA given an identity cutoff. Typical values for the cutoff are 0.8.
    """
    from scipy import sparse as sp
    msa_1hot = msa_1hot.reshape(msa_1hot.shape[0],-1)
    msa_1hot_sp = sp.csr_matrix(msa_1hot)
    id_mtx = msa_1hot_sp @ (msa_1hot_sp.T @ sp.csr_matrix(np.ones(msa_1hot.shape[0])).T)
    w = 1.0 / id_mtx.toarray().squeeze()
    return w


def weight_msa_vectorized(msa_1hot, cutoff):
    """
    Finds the weight for each MSA given an identity cutoff. Typical values for the cutoff are 0.8.
    NOTE that this will take a long time for MSAs with a lot of sequences and might be possible to speed up by using generators or something similar.
    The reason why this takes such a long time is that tensordot returns a (n x n) matrix where n is the number of sequences in the MSA.
    Follows the procedures used by trRossetta
    """
    id_min = msa_1hot.shape[1] * cutoff
    w = np.empty(msa_1hot.shape[0])
    for i in range(msa_1hot.shape[0]):
        tmp = msa_1hot[i, :, :]
        id_mtx = np.tensordot(msa_1hot, tmp[None,:,:], axes=([1, 2], [1, 2]))
        id_mask = id_mtx > id_min
        w[i] = 1.0 / np.sum(id_mask)
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




class switch(object):
    """Switch statement for Python, based on recipe from Python Cookbook."""

    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:  # changed for v1.5
            self.fall = True
            return True
        else:
            return False


def letter_to_num(string, dict_):
    """ Convert string of letters to list of ints """
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')
    num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
    num = [int(i) for i in num_string.split()]
    return num

def letter_to_bool(string, dict_):
    """ Convert string of letters to list of bools """
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')
    num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
    num = [bool(int(i)) for i in num_string.split()]
    return num



def read_record(file_, num_evo_entries):
    """ Read all protein records from pnet file. """
    id = []
    seq = []
    pssm = []
    entropy = []
    dssp = []
    coord = []
    mask = []
    seq_len = []
    scaling = 0.001 # converts from pico meters to nanometers

    t0 = time.time()
    while True:
        next_line = file_.readline()
        for case in switch(next_line):
            if case('[ID]' + '\n'):
                id.append(file_.readline()[:-1])
                if len(id) % 1000 == 0:
                    print("loading sample: {:}, Time: {:2.2f}".format(len(id),time.time() - t0))
            elif case('[PRIMARY]' + '\n'):
                seq.append(letter_to_num(file_.readline()[:-1], AA_DICT))
                seq_len.append(len(seq[-1]))
            elif case('[EVOLUTIONARY]' + '\n'):
                evolutionary = []
                for residue in range(num_evo_entries):
                    evolutionary.append([float(step) for step in file_.readline().split()])
                # pssm.append(evolutionary)
                entropy_i = [float(step) for step in file_.readline().split()]
                # entropy.append(entropy_i)
            elif case('[SECONDARY]' + '\n'):
                dssp_i = letter_to_num(file_.readline()[:-1], DSSP_DICT)
                # dssp.append(dssp_i)
            elif case('[TERTIARY]' + '\n'):
                tertiary = []
                for axis in range(NUM_DIMENSIONS):
                    tertiary_i = [float(coord)*scaling for coord in file_.readline().split()]
                    tertiary.append(tertiary_i)
                coord.append(tertiary)
            elif case('[MASK]' + '\n'):
                mask_i = letter_to_bool(file_.readline()[:-1], MASK_DICT)
                # mask.append()
            elif case(''):


                return id,seq,pssm,entropy,dssp,coord,mask,seq_len

def parse_pnet_for_comparison(file,max_seq_len=-1):
    with open(file, 'r') as f:
        t0 = time.time()
        id, seq, pssm, entropy, dssp, coords, mask, seq_len = read_record(f, 20)
        #NOTE THAT THE RESULT IS RETURNED IN ANGSTROM
        print("loading data complete! Took: {:2.2f}".format(time.time()-t0))
        r1 = []
        r2 = []
        r3 = []
        pssm2 = []
        for i in range(len(coords)): #We transform each of these, since they are inconveniently stored
        #     pssm2.append(flip_multidimensional_list(pssm[i]))
        #     # Note that we are changing the order of the coordinates, as well as which one is first, since we want Carbon alpha to be the first, Carbon beta to be the second and Nitrogen to be the third
            r1.append((separate_coords(coords[i], 1)))
            r2.append((separate_coords(coords[i], 2)))
            r3.append((separate_coords(coords[i], 0)))
        #
        #     if i+1 % 1000 == 0:
        #         print("flipping and separating: {:}, Time: {:2.2f}".format(len(id), time.time() - t0))

        convert = ListToNumpy()
        seq = convert(seq)
        r1 = convert(r1)
        r2 = convert(r2)
        r3 = convert(r3)
        seq_len = np.array(seq_len)

        print("parse complete! Took: {:2.2f}".format(time.time() - t0))
    return seq, seq_len, id, r1,r2,r3


def sparse_one_hot_encoding(data,cat=21):
    import scipy.sparse as scsp
    n,l = data.shape
    i = np.arange(n)
    ii = np.repeat(i[:, None], l, axis=1)
    a = np.arange(0, l * cat, cat)
    b = a[None, :]
    jj = b + data

    v = np.ones_like(ii)
    onehot = scsp.coo_matrix((v.flatten(), (ii.flatten(), jj.flatten())), dtype=np.float32)
    return scsp.csr_matrix(onehot)

if __name__ == "__main__":
    import sparse as sp
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')  # TkAgg



    # MSA_folder = "F:/Globus/raw/"
    MSA_folder = "F:/Globus/raw_subset/"
    avg_read_time = 0
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
            indices = np.random.permutation(n)
            pssms = []
            f2d_dcas = []
            for j in range(max(min(n_repeats,n//max_samples),1)):
                idx = indices[j*max_samples:min((j+1)*max_samples,n)]
                msa = msas[idx,:]
                x_sp = sparse_one_hot_encoding(msa)

                t1 = time.time()
                nsub = min(nsub_size,n)
                w = ANN_sparse(x_sp[0:nsub], x_sp, k=100, eff=300, cutoff=True)
                mask = w == np.min(w)
                wm = np.sum(w[mask])
                if wm > 0.01 * np.sum(w): # If these account for more than 10% of the total weight, we scale them down to almost 10%.
                    scaling = wm/np.sum(w)/0.01
                else:
                    scaling = 1
                w[mask] /= scaling
                t2 = time.time()

                wn = w/np.sum(w)
                # if n > nsamples:
                #     idx = np.random.choice(n, nsamples, replace=False, p=wn)
                #     ws = w[idx]
                #     msa_1hot = np.eye(21, dtype=np.float32)[msa[idx,:]]
                # else:
                #     ws = w
                msa_1hot = np.eye(21, dtype=np.float32)[msa]

                pssms.append(msa2pssm(msa_1hot, w))
                f2d_dcas.append(dca(msa_1hot, w))
                t3 = time.time()

                print("# MSAs = {:}, time taken: {:2.2f}, {:2.2f}, {:2.2f}".format(n, t1 - t0, t2 - t1, t3 - t2))
        f2d_dca = np.mean(np.asarray(f2d_dcas),axis=0)
        pssm = np.mean(np.asarray(pssms),axis=0)

        plt.imshow(f2d_dcas[0][:,:,-1])
        plt.figure()
        plt.imshow(f2d_dcas[1][:,:,-1])
        plt.figure()
        plt.imshow(f2d_dca[:,:,-1])
        # plt.figure()
        # plt.imshow(f2d_dcas[2][:,:,-1])
        # plt.figure()
        # plt.imshow(f2d_dcas[3][:,:,-1])
        # plt.figure()
        # plt.imshow(f2d_dcas[4][:, :, -1])
        # plt.figure()
        # plt.imshow(f2d_dcas[5][:, :, -1])
        # plt.figure()
        # plt.imshow(f2d_dcas[6][:, :, -1])
        # plt.figure()
        # plt.imshow(f2d_dcas[7][:, :, -1])
        # plt.figure()
        # plt.imshow(f2d_dcas[8][:, :, -1])
        plt.pause(300)
        print("look")


    #
    # # This second part is meant to be the real deal
    # # Here we load the full protein net training set,
    # # and then go through the MSA set iteratively, and compare them against the protein net set.
    # pnet_file = "./../data/training_30.pnet"
    # # pnet_file = "./../data/testing.pnet"
    #
    # t0 = time.time()
    # seqs, seqs_len, id,r1,r2,r3 = parse_pnet_for_comparison(pnet_file, max_seq_len=-1)
    #
    # t1 = time.time()
    # print("Read the pnet_file, took {:2.2f}".format(t1-t0))
    # seqs_len_unique, counts = np.unique(seqs_len, return_counts=True)
    # a=np.digitize(seqs_len,bins=seqs_len_unique,right=True)
    # lookup = {}  # create an empty dictionary
    # # Make a lookup table such that t[seq_len] = idx
    # for i, seq_len_unique in enumerate(seqs_len_unique):
    #     lookup[seq_len_unique] = i
    #
    # # Next we create the list of arrays, and then afterwards we populate them
    # seqs_list = []
    # seqs_list_org_id = []
    # for seq_len_unique,count in zip(seqs_len_unique,counts):
    #     tmp = np.empty((count,seq_len_unique), dtype=np.int32)
    #     seqs_list.append(tmp)
    #     tmp = np.empty((count,1), dtype=np.int32)
    #     seqs_list_org_id.append(tmp)
    #
    # counter = np.zeros_like(counts)
    # for i,(seq,seq_len) in enumerate(zip(seqs,seqs_len)):
    #     seq_idx = lookup[seq_len]
    #     counter_idx = counter[seq_idx]
    #     seqs_list[seq_idx][counter_idx,:] = seq
    #     seqs_list_org_id[seq_idx][counter_idx] = i
    #     counter[seq_idx] += 1
    #
    # n = len(seqs)
    # matches = np.zeros(n, dtype=np.int32)
    #
    # # for i,(seq,seq_len) in enumerate(zip(seqs,seqs_len)):
    # #     seq_idx = lookup[seq_len]
    # #     seqs_i = seqs_list[seq_idx]
    # #     res = np.mean(seq == seqs_i,axis=1) == 1
    # #     if np.sum(res) == 1:
    # #         continue
    # #     else:
    # #         id = np.where(res == True)[0]
    # #         print("{:}, in {:}, {:}".format(i,seq_idx,id))
    # #         print("what now?")
    #
    #
    #
    #
    # # I think the correct way to do this is to sort pnet, by length of the proteins, and group all proteins with the same length in an nd.array
    # # Make a lookup table with idx -> seq_len
    # # Then the comparison is only against all elements in that particular group.
    # t2 = time.time()
    # print("Built the data structure, took {:2.2f}".format(t2-t1))
    #
    # n_matches = 0
    # MSA_not_in_protein_net = 0
    # MSA_in_protein_net_more_than_once = 0
    #
    # MSA_folder = "F:/Globus/raw/"
    # avg_read_time = 0
    # # MSA_folder = "F:/Globus/raw_subset/"
    # search_command = MSA_folder + "*.a2m.gz"
    # a2mfiles = [f for f in glob.glob(search_command)]
    # for i,a2mfile in enumerate(a2mfiles):
    #     tt0 = time.time()
    #     msa = read_a2m_gz_file(a2mfile)
    #     tt1 = time.time()
    #     avg_read_time += tt1-tt0
    #     # print("Read: {:2.2f}, Average: {:2.2f}".format(tt1-tt0, avg_read_time/(i+1)))
    #     protein = msa[-1,:]
    #     seq_len = len(protein)
    #     try:
    #         seq_idx = lookup[seq_len]
    #     except:
    #         MSA_not_in_protein_net += 1
    #         continue
    #     seqs_i = seqs_list[seq_idx]
    #     res = np.mean(protein == seqs_i,axis=1) == 1
    #     tt2 = time.time()
    #     if np.sum(res) == 1:
    #         n_matches += 1
    #         org_id = seqs_list_org_id[seq_idx][res]
    #         matches[org_id] += 1
    #
    #
    #
    #
    #     elif np.sum(res) == 0:
    #         MSA_not_in_protein_net += 1
    #     else:
    #         MSA_in_protein_net_more_than_once += 1
    #     if (i+1) % 1000 == 0:
    #         t25 = time.time()
    #         print("Compared {:} proteins. Matches: {:}, MSA not in pnet: {:}, MSAs in pnet more than once {:}. Time taken: {:2.2f} ".format(i+1,n_matches,MSA_not_in_protein_net,MSA_in_protein_net_more_than_once,t25-t2))
    #
    # t3 = time.time()
    # print("Ran the comparisons, took {:2.2f}".format(t3-t2))

print("done")