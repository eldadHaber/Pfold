import time
import torch
import numpy as np
from xopen import xopen
import numpy as np
import string

from ML_MSA.MSA_utils import AA_converter


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
    # convert letters into numbers
    msa = AA(seqs)
    # alphabet = np.array(AA_LIST, dtype='|S1').view(np.uint8)
    # msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)

    # for i in range(alphabet.shape[0]):
    #     msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    # msa[msa > 20] = 20
    return msa
#
# def load_proteinnet_msas_and_cov(a2mfile,n_repeats=3,max_samples=10000,device='cpu'):
#     msas = read_a2m_gz_file(a2mfile, verbose=True)
#
#     protein = msas[-1, :]
#     n = msas.shape[0]
#     indices = np.random.permutation(n)
#     pssms = []
#     f2d_dcas = []
#     for j in range(max(min(n_repeats, n // max_samples), 1)):
#         idx = indices[j * max_samples:min((j + 1) * max_samples, n)]
#         msa = msas[idx, :]
#         x_sp = sparse_one_hot_encoding(msa)
#
#         nsub = min(nsub_size, n)
#         w = ANN_sparse(x_sp[0:nsub], x_sp, k=100, eff=300, cutoff=True)
#         mask = w == np.min(w)
#         wm = np.sum(w[mask])
#         if wm > 0.01 * np.sum(
#                 w):  # If these account for more than 1% of the total weight, we scale them down to almost 1%.
#             scaling = wm / np.sum(w) / 0.01
#         else:
#             scaling = 1
#         w[mask] /= scaling
#
#         wn = w / np.sum(w)
#         msa_1hot = np.eye(21, dtype=np.float32)[msa]
#         t5 = time.time()
#         t.ann_time.add_time(t5 - t4)
#         pssms.append(msa2pssm(msa_1hot, w))
#         f2d_dcas.append(dca(msa_1hot, w))
#         t6 = time.time()
#         t.dca_time.add_time(t6 - t5)
#     f2d_dca = np.mean(np.asarray(f2d_dcas), axis=0)
#     pssm = np.mean(np.asarray(pssms), axis=0)
#
#     # SAVE FILE HERE
#     if IDs is None:
#         fullfileout = "{:}ID_{:}".format(outputfolder, i)
#     else:
#         fullfileout = "{:}ID_{:}".format(outputfolder, IDs[i])
#     np.savez(fullfileout, protein=protein, pssm=pssm, dca=f2d_dca, r1=r1[org_id].T, r2=r2[org_id].T, r3=r3[org_id].T)
#
#     elif np.sum(res) == 0:
#         c.MSAs_no_match += 1
#     else:
#         c.MSAs_multi_match += 1
#     if (i + 1) % report_freq == 0:
#         print(
#             "Compared {:} proteins. Matches: {:}, MSA not in pnet: {:}, MSAs in pnet more than once {:}, excluded: {:}, Avr Time(read): {:2.2f}, Avr Time(lookup): {:2.2f}, Avr Time(ANN): {:2.2f}, Avr Time(DCA): {:2.2f}, Total Time(read): {:2.2f}, Total Time(lookup): {:2.2f}, Total Time(ANN): {:2.2f}, Total Time(DCA): {:2.2f} Time(total): {:2.2f}".format(
#                 i + 1, c.match, c.MSAs_no_match, c.MSAs_multi_match, c.excluded, t.read_time(), t.lookup_time(),
#                 t.ann_time(), t.dca_time(), t.read_time(total=True), t.lookup_time(total=True), t.ann_time(total=True),
#                 t.dca_time(total=True), time.time() - t.t0))
#
#     # finish iterating through dataset
#     print(
#         "Compared {:} proteins. Matches: {:}, MSA not in pnet: {:}, MSAs in pnet more than once {:}, excluded: {:}, Avr Time(read): {:2.2f}, Avr Time(lookup): {:2.2f}, Avr Time(ANN): {:2.2f}, Avr Time(DCA): {:2.2f}, Total Time(read): {:2.2f}, Total Time(lookup): {:2.2f}, Total Time(ANN): {:2.2f}, Total Time(DCA): {:2.2f} Time(total): {:2.2f}".format(
#             i + 1, c.match, c.MSAs_no_match, c.MSAs_multi_match, c.excluded, t.read_time(), t.lookup_time(),
#             t.ann_time(), t.dca_time(), t.read_time(total=True), t.lookup_time(total=True), t.ann_time(total=True),
#             t.dca_time(total=True), time.time() - t.t0))
#

