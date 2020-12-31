import torch
import numpy as np

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

def msa2weight(msa_1hot, cutoff=0.8):
    """
    Finds the weight for each MSA given an identity cutoff. Typical values for the cutoff are 0.8.
    NOTE that this will take a long time for MSAs with a lot of sequences and might be possible to speed up by using generators or something similar.
    The reason why this takes such a long time is that tensordot returns a (n x n) matrix where n is the number of sequences in the MSA.
    Follows the procedures used by trRossetta
    """
    assert cutoff < 1
    import time
    L = msa_1hot.shape[1]
    id_min = L * cutoff
    id_mtx = torch.tensordot(msa_1hot, msa_1hot, dims=([1, 2], [1, 2]))
    S = id_mtx > id_min
    id_mtx = None
    t1 = time.time()
    S = S.to('cpu')
    S = torch.sum(S, dim=-1)
    S = S.to(device=msa_1hot.device)
    t2 = time.time()
    print("time to compute S {:2.2f}".format(t2-t1))
    w = 1.0 / S.to(dtype=torch.float16)
    Neff = torch.sum(w)
    Nf = Neff/np.sqrt(L)
    return w, Nf

def msa2pssm(msa_1hot, w):
    """
    Computes the position scoring matrix, f_i, given an MSA and its weight.
    Furthermore computes the sequence entropy, h_i.
    Follows the procedures used by trRossetta
    """
    neff = torch.sum(w)
    pssm = torch.sum(w[:, None, None] * msa_1hot, dim=0) / neff
    entropy = torch.sum(- (pssm + 1e-9) * torch.log(pssm + 1e-9), dim=1)
    return pssm, entropy

def msa2cov(msa_1hot, w, penalty=4.5):
    """
    This follows the procedures used by trRossetta.
    Computes the covariance and inverse covariance matrix (equation 2), as well as the APC (equation 4).
    """

    nr, nc, ns = msa_1hot.shape
    x = msa_1hot.reshape(nr, nc * ns)
    num_points = torch.sum(w) - torch.sqrt(torch.mean(w))
    mean = torch.sum(x * w[:, None], dim=0, keepdims=True) / num_points
    x = (x - mean) * torch.sqrt(w[:, None])
    cov = torch.matmul(x.T, x) / num_points
    cov_reg = cov + torch.eye(nc * ns,device=cov.device) * penalty / torch.sqrt(torch.sum(w))
    cov = None
    inv_cov = torch.inverse(cov_reg)
    x1 = inv_cov.reshape(nc, ns, nc, ns)
    x2 = x1.permute(0,2,1,3)
    features = x2.reshape(nc, nc, ns * ns)

    x3 = torch.sqrt(torch.sum(torch.square(x1[:, :-1, :, :-1]), dim=(1,3))) * (1 - torch.eye(nc, device=x1.device))
    apc = torch.sum(x3, dim=0, keepdims=True) * torch.sum(x3, dim=1, keepdims=True) / torch.sum(x3)
    contacts = (x3 - apc) * (1 - torch.eye(nc,device=x3.device))
    return features, contacts


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

