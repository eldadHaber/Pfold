import time
import numpy as np
import glob

from preprocessing.ANN import ANN_sparse
from preprocessing.MSA_reader import read_a2m_gz_file
from srcOld.dataloader_pnet import parse_pnet

from srcOld.utils import Timer


def setup_protein_comparison(seqs, seqs_len):
    """
    This function will take a list of sequences and a corresponding array of sequence lengths and create the variables needed for a fast matching of any future protein to the proteins in these.
    The code works by grouping all sequences of similar size together in an array. Hence when a new protein needs to be compared it is only compared against all the proteins of similar size rather than all the proteins.
    Future optimizations could of course be done to improve this even more, but for now this is fast enough.

    The way a comparison should be done is by looking up the sequence length in the lookup table, and if it exist then do a comparison against all sequences in the returned array.

    A small code example of how to use this to look up a new protein:

    ____________________________________________________________________________________
    new_protein = read_from_file(...)
    seq_len = len(new_protein)

    Try:
        seq_idx = lookup[seq_len]
    except:
        print("Protein length not in database")
        continue
    seqs_i = seqs_list[seq_idx]
    res = np.mean(protein == seqs_i, axis=1) == 1
    org_id = (seqs_list_org_id[seq_idx][res]).squeeze()
    _____________________________________________________________________________________

    so seqs_i is all the sequences with the same length as protein.
    and org_id is the id of the sequence that match protein.
    """
    seqs_len_unique, counts = np.unique(seqs_len, return_counts=True)
    lookup = {}  # create an empty dictionary
    # Make a lookup table such that t[seq_len] = idx
    for i, seq_len_unique in enumerate(seqs_len_unique):
        lookup[seq_len_unique] = i

    # Next we create the list of arrays, and then afterwards we populate them
    seqs_list = []
    seqs_list_org_id = []
    for seq_len_unique, count in zip(seqs_len_unique, counts):
        tmp = np.empty((count, seq_len_unique), dtype=np.int32)
        seqs_list.append(tmp)
        tmp = np.empty((count, 1), dtype=np.int32)
        seqs_list_org_id.append(tmp)

    # Populate the arrays
    counter = np.zeros_like(counts)
    for i, (seq, seq_len) in enumerate(zip(seqs, seqs_len)):
        seq_idx = lookup[seq_len]
        counter_idx = counter[seq_idx]
        seqs_list[seq_idx][counter_idx, :] = seq
        seqs_list_org_id[seq_idx][counter_idx] = i
        counter[seq_idx] += 1
    return seqs_list, seqs_list_org_id, lookup



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

class Timers:
    def __init__(self):
        self.read_time = Timer()
        self.lookup_time = Timer()
        self.ann_time = Timer()
        self.dca_time = Timer()
        self.t0 = time.time()
        return

class Counters:
    def __init__(self):
        self.match = 0
        self.MSAs_no_match = 0
        self.MSAs_multi_match = 0
        self.excluded = 0


if __name__ == "__main__":
    pnet_file = "./data/testing.pnet"
    MSA_folder = "./data/MSA/"
    outputfolder = "./data/cov/"
    max_seq_len = 10000
    min_seq_len = 80
    write_freq = 2
    report_freq = 5
    max_samples = 20000
    nsub_size = 8000
    n_repeats = 3
    min_msas = 0

    t = Timers()
    c = Counters()

    t0 = time.time()
    args = parse_pnet(pnet_file, max_seq_len=max_seq_len, min_seq_len=min_seq_len, use_dssp=False, use_pssm=False, use_mask=False, use_entropy=False)
    seqs = args['seq']
    seqs_len = args['seq_len']
    r1 = args['r1']
    r2 = args['r2']
    r3 = args['r3']
    n_samples = len(seqs_len)
    print("Read the pnet_file, took {:2.2f}s, contains {:} samples".format(time.time() - t0, n_samples))

    seqs_list, seqs_list_org_id, lookup = setup_protein_comparison(seqs, seqs_len)

    matches = np.zeros(n_samples, dtype=np.int32)

    search_command = MSA_folder + "*.a2m.gz"
    a2mfiles = [f for f in glob.glob(search_command)]

    # We search the outputfolder to see whether there already exists some files, if there do, we find the largest number and start reading the MSAs from that number onwards.
    search_command = outputfolder + "*.npz"
    outputfiles = [f for f in glob.glob(search_command)]
    ite_start = -1
    for outputfile in outputfiles:
        str_tmp = outputfile.split(sep="ID")[-1]
        num = int("".join(filter(str.isdigit, str_tmp)))
        ite_start = max(num,ite_start)


    for i, a2mfile in enumerate(a2mfiles):
        if i <= ite_start:
            continue
        t1 = time.time()
        msas = read_a2m_gz_file(a2mfile,max_seq_len=max_seq_len,min_seq_len=min_seq_len, verbose=True)

        if msas is None or msas.shape[0] < min_msas:
            c.excluded += 1
            continue
        t2 = time.time()
        t.read_time.add_time(t2-t1)

        protein = msas[-1, :]
        seq_len = len(protein)
        try:
            seq_idx = lookup[seq_len]
        except:
            c.MSAs_no_match += 1
            continue
        seqs_i = seqs_list[seq_idx]
        res = np.mean(protein == seqs_i, axis=1) == 1
        t3 = time.time()
        t.lookup_time.add_time(t3-t2)
        if np.sum(res) == 1:
            c.match += 1
            org_id = (seqs_list_org_id[seq_idx][res]).squeeze()
            matches[org_id] += 1
            r1i = r1[org_id].T
            r2i = r2[org_id].T
            r2i = r3[org_id].T
            # print("msa shape {:}, r1 shape {:} ".format(msa.shape,r1i.shape))

            n = msas.shape[0]
            l = msas.shape[1]
            indices = np.random.permutation(n)
            pssms = []
            f2d_dcas = []
            for j in range(max(min(n_repeats, n // max_samples), 1)):
                t4 = time.time()
                idx = indices[j * max_samples:min((j + 1) * max_samples, n)]
                msa = msas[idx, :]
                x_sp = sparse_one_hot_encoding(msa)

                nsub = min(nsub_size, n)
                w = ANN_sparse(x_sp[0:nsub], x_sp, k=100, eff=300, cutoff=True)
                mask = w == np.min(w)
                wm = np.sum(w[mask])
                if wm > 0.01 * np.sum(w):  # If these account for more than 1% of the total weight, we scale them down to almost 1%.
                    scaling = wm / np.sum(w) / 0.01
                else:
                    scaling = 1
                w[mask] /= scaling

                wn = w / np.sum(w)
                msa_1hot = np.eye(21, dtype=np.float32)[msa]
                t5 = time.time()
                t.ann_time.add_time(t5-t4)
                pssms.append(msa2pssm(msa_1hot, w))
                f2d_dcas.append(dca(msa_1hot, w))
                t6 = time.time()
                t.dca_time.add_time(t6-t5)
            f2d_dca = np.mean(np.asarray(f2d_dcas), axis=0)
            pssm = np.mean(np.asarray(pssms), axis=0)

            # SAVE FILE HERE
            fullfileout = "{:}ID_{:}".format(outputfolder,i)
            np.savez(fullfileout,protein=protein,pssm=pssm,dca=f2d_dca,r1=r1[org_id].T,r2=r2[org_id].T,r3=r3[org_id].T)

        elif np.sum(res) == 0:
            c.MSAs_no_match += 1
        else:
            c.MSAs_multi_match += 1
        if (i + 1) % report_freq == 0:
            print("Compared {:} proteins. Matches: {:}, MSA not in pnet: {:}, MSAs in pnet more than once {:}, excluded: {:}, Avr Time(read): {:2.2f}, Avr Time(lookup): {:2.2f}, Avr Time(ANN): {:2.2f}, Avr Time(DCA): {:2.2f}, Total Time(read): {:2.2f}, Total Time(lookup): {:2.2f}, Total Time(ANN): {:2.2f}, Total Time(DCA): {:2.2f} Time(total): {:2.2f}".format(
                  i + 1, c.match, c.MSAs_no_match, c.MSAs_multi_match, c.excluded, t.read_time(),t.lookup_time(),t.ann_time(),t.dca_time(), t.read_time(total=True),t.lookup_time(total=True),t.ann_time(total=True),t.dca_time(total=True),time.time()-t.t0))

    # finish iterating through dataset
    print(
        "Compared {:} proteins. Matches: {:}, MSA not in pnet: {:}, MSAs in pnet more than once {:}, excluded: {:}, Avr Time(read): {:2.2f}, Avr Time(lookup): {:2.2f}, Avr Time(ANN): {:2.2f}, Avr Time(DCA): {:2.2f}, Total Time(read): {:2.2f}, Total Time(lookup): {:2.2f}, Total Time(ANN): {:2.2f}, Total Time(DCA): {:2.2f} Time(total): {:2.2f}".format(
            i + 1, c.match, c.MSAs_no_match, c.MSAs_multi_match, c.excluded, t.read_time(), t.lookup_time(),
            t.ann_time(), t.dca_time(), t.read_time(total=True), t.lookup_time(total=True), t.ann_time(total=True),
            t.dca_time(total=True), time.time() - t.t0))
