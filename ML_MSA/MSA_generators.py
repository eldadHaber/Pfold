import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class DummyModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, alphabet,unknown_idx):
        super(DummyModel, self).__init__()
        self.alphabet = alphabet
        self.n = len(alphabet)
        self.unknown_idx = unknown_idx

    def forward(self, seq):
        mask = seq == self.unknown_idx
        tmp = seq.clone()
        mm = tmp == self.unknown_idx
        tmp[mm] = 0
        onehot = F.one_hot(tmp.long(), num_classes=self.n,).to(dtype=torch.float32)
        nb,l = seq.shape
        for i in range(nb):
            for j in range(l):
                if seq[i,j] == self.unknown_idx:
                    tt = torch.rand(len(self.alphabet))
                    tt = tt / torch.sum(tt)
                    onehot[i,j,:] = tt
        return onehot


class ModelWrapper(nn.Module):
    """
    This is a wrapper around a pretrained model.
    This kind of wrapper is usefull if the pretrained model, uses a different alphabet than the one you wish your output in,
    or if the model has some prepending/appending tokens, that needs to be applied to any sentence before model evaluation.
    """

    def __init__(self, model, translator):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.translator = translator

    def forward(self, seqs):
        """
        seqs has shape (nb,n)
        """
        seqs_translated = self.translator.forward(seqs)
        results = self.model(seqs_translated)
        pred_facebook = results['logits']
        pred_pnet = self.translator.backward_probability_dist(pred_facebook)
        prob_pnet = torch.softmax(pred_pnet, dim=-1)

        return prob_pnet



def MSA_iter(seqs,mask_id=32):
    ite = 0
    batch_idx = torch.arange(n)
    MSA_iter = seqs.clone()
    mask = seqs == mask_id
    with torch.no_grad():
        while True:
            t0 = time.time()
            results = model(seqs)
            t1 = time.time()
            pred = results['logits']
            prob = torch.softmax(pred, dim=2)

            t2 = time.time()
            val, idx = torch.max(prob, dim=2)
            nm = torch.sum(mask, dim=1)
            nmtot = torch.sum(mask)
            t3 = time.time()

            if torch.sum(nm == 0) > 0:
                MSA_iter[batch_idx[nm == 0], :] = seqs[nm == 0, :]
            t4 = time.time()

            if nmtot == 0:
                break
            val = val[nm > 0, :]
            idx = idx[nm > 0, :]
            mask = mask[nm > 0, :]
            t45 = time.time()
            seqs = seqs[nm > 0, :]
            batch_idx = batch_idx[nm > 0]
            t5 = time.time()

            nremain = seqs.shape[0]

            val_m = val
            val_m[mask == False] = 0

            next_aa = torch.argmax(val_m, dim=1)
            t6 = time.time()

            nn = torch.arange(nremain)
            mask[nn, next_aa] = False
            seqs[nn, next_aa] = idx[nn, next_aa].to(device)
            t7 = time.time()
            ite += 1
        MSA_iter = MSA_iter[:,1:-1]
    return MSA_iter

def MSA_once(seqs,model,mask_id=32,deterministic=True):
    mask = seqs == mask_id
    with torch.no_grad():
        results = model(seqs)
        pred = results['logits']
        if deterministic:
            aa_select = torch.argmax(pred, dim=2)
        else:
            u = torch.rand((pred.shape[0],pred.shape[1]),device=pred.device)
            prob = torch.softmax(pred, dim=2)

            aa_select = (torch.cumsum(prob,dim=2) < u[:,:,None]).sum(dim=2)
        seqs[mask] = aa_select[mask]
    return seqs


def MSA_once(seqs,model,mask_id=32,deterministic=True):
    mask = seqs == mask_id
    with torch.no_grad():
        results = model(seqs)
        pred = results['logits']
        if deterministic:
            aa_select = torch.argmax(pred, dim=2)
        else:
            u = torch.rand((pred.shape[0],pred.shape[1]),device=pred.device)
            prob = torch.softmax(pred, dim=2)

            aa_select = (torch.cumsum(prob,dim=2) < u[:,:,None]).sum(dim=2)
        seqs[mask] = aa_select[mask]
    return seqs




def mask_scheme1(seq_tok,n=10,p_mask=0.2,mask_id=32,prepend_idx=0,append_idx=2):
    """This will o nly work with the append prepend idx used"""
    n_added_tokens = 0
    if prepend_idx >= 0:
        n_added_tokens += 1
    if append_idx >= 0:
        n_added_tokens += 1
    l = seq_tok.shape[-1] + n_added_tokens
    rng = np.random.default_rng()
    pm = rng.random((n,l))
    seq_rep = seq_tok.repeat(n,1)
    seq_batch = torch.empty(n,l,dtype=torch.int64,device=seq_tok.device)
    seq_batch[:,1:-1] = seq_rep
    idx = torch.from_numpy(pm < p_mask)
    seq_batch[idx] = mask_id
    seq_batch[:,0] = 0 #Start token
    seq_batch[:,-1] = 2 #End token
    mask = seq_batch == mask_id
    return seq_batch, mask


def mask_scheme_pssm(seq_tok,mask_id=32,prepend_idx=0,append_idx=2):
    """This will o nly work with the append prepend idx used"""
    n = seq_tok.shape[-1]
    n_added_tokens = 0
    if prepend_idx >= 0:
        n_added_tokens += 1
    if append_idx >= 0:
        n_added_tokens += 1
    l = n + n_added_tokens
    seq_rep = seq_tok.repeat(n,1)
    seq_batch = torch.empty(n,l,dtype=torch.int64,device=seq_tok.device)
    seq_batch[:,1:-1] = seq_rep
    idx = torch.zeros((n,l),dtype=torch.bool)
    ii = torch.arange(n)
    jj = torch.arange(start=1,end=n+1)
    idx[ii,jj] = True
    seq_batch[idx] = mask_id
    seq_batch[:,0] = 0 #Start token
    seq_batch[:,-1] = 2 #End token
    mask = seq_batch == mask_id
    return seq_batch, mask

def mask_scheme_cov(seq_tok,test_toks=[4,5,6],mask_id=32,prepend_idx=0,append_idx=2):
    """This will o nly work with the append prepend idx used"""
    n = seq_tok.shape[-1]
    n_added_tokens = 0
    if prepend_idx >= 0:
        n_added_tokens += 1
    if append_idx >= 0:
        n_added_tokens += 1
    l = n + n_added_tokens
    nt = len(test_toks)
    nc = n*(n-1)
    nb = nc*nt
    seq_rep = seq_tok.repeat(nb,1)
    seq_batch = torch.empty(nb,l,dtype=torch.int64,device=seq_tok.device)
    seq_batch[:,1:-1] = seq_rep
    # seq_batch[:,:] = seq_rep
    idx = torch.zeros((nb,l),dtype=torch.bool)
    ni = torch.arange(nb)
    ii = (torch.arange(start=1,end=n+1)).repeat(n).repeat(nt)
    jj = torch.arange(start=1,end=n+1).repeat_interleave(n).repeat(nt)
    tok_i = torch.tensor(test_toks).repeat_interleave(n*(n-1))

    mm = ii != jj
    ii = ii[mm]
    jj = jj[mm]
    seq_batch[ni,ii] = mask_id
    seq_batch[ni,jj] = tok_i
    seq_batch[:,0] = 0 #Start token
    seq_batch[:,-1] = 2 #End token
    # mask = seq_batch == mask_id
    return seq_batch, ii,jj,tok_i

def compute_reference_matrix(model,seqs,idxrow,idxcol,maxresidues,device,ncategories):
    """
    This runs a series a sequences (seqs) through the neural network (model),
    and saves the selected predicted probabilities according to idxrow, idxcol
    into a matrix M (l,l,ncategories)
    M is ordered such that the first dimension determines the residue probability shown, while the second index gives the accompanying masked index.
    Hence:
        M[2,5,:] corresponds to a sequence, where the 2nd and 5th residue was masked, and the probabilities shown in M[2,5,:] corresponds to the 2nd residue, while M[5,2,:] corresponds to the 5th residue,
    """
    nb,l = seqs.shape
    assert maxresidues >= l,  "The protein is longer than the maximum number of residues allowed at once."
    seq_pr_iteration = int(np.floor(maxresidues/l))
    niters = int(np.ceil(nb/seq_pr_iteration)) # Compute the number of iterations we will have to split the job into in order to calculate it.
    M = torch.zeros((l,l,ncategories),dtype=torch.float32,device=device)

    for i in range(niters):
        i0 = i*seq_pr_iteration
        i1 = (i+1)*seq_pr_iteration
        idxrowi = idxrow[i0:i1]
        idxcoli = idxcol[i0:i1]
        seq_ref_batch = seqs[i0:i1,:]
        with torch.no_grad():
            prob = model(seq_ref_batch)
            tmp = torch.arange(idxrowi.shape[0], device=device)
            pselect = prob[tmp,idxrowi,:]
            M[idxrowi,idxcoli,:] = pselect
            pselect = prob[tmp,idxcoli,:]
            M[idxcoli,idxrowi,:] = pselect
    return M

def setup_reference_matrix_job(seq_tok,mask_id=32):
    """
    Takes a numeric sequence (n), and outputs a numeric batch of sequences (nb,n).
    Each sequence in the sequence batch matches a unique pair masking of the original sequence.
    Hence the number of sequence in the bacth, is nb=n*(n-1)/2
    The routine furthermore returns ii,jj which corresponds masking indices.
    """
    n = seq_tok.shape[-1]
    nc = n*(n-1)//2
    seq_batch = seq_tok.repeat(nc,1)
    ni = torch.arange(nc)
    ii = (torch.arange(start=0,end=n)).repeat(n)
    jj = torch.arange(start=0,end=n).repeat_interleave(n)
    mm = ii > jj
    ii = ii[mm]
    jj = jj[mm]
    seq_batch[ni,ii] = mask_id
    seq_batch[ni,jj] = mask_id
    return seq_batch, ii, jj


def setup_conditional_prob_matrix_job(seq_tok,alphabet=['A','B','C'],mask_id=32):
    """
    Takes a numeric sequence (seq_tok) of length n, and outputs a numeric batch of sequences (seq_batch) of shape (nb,n).
    The routine goes through seq_tok and takes any possible residue pair combination. For each pair it masks one of them while setting the other residue to each letter in the alphabet converted to numerical values.
    This creates nb=n*(n-1)*len(alphabet) sequences.
    Along with the batch sequences is outputted, maskids, targetids, tokenids.
    Where:
        maskids details which residue was masked in a given batch sequence.
        targetids details which residue was altered to another letter in the alphabet.
        tokenids details what letter the target was altered to.
    """

    n = seq_tok.shape[-1]
    nt = len(alphabet)
    nc = n*(n-1)
    nb = nc*nt
    alphabet_num = list(range(nt))
    seq_rep = seq_tok.repeat(nb,1)
    seq_batch = torch.empty(nb,n,dtype=torch.int64,device=seq_tok.device)
    seq_batch[:,:] = seq_rep
    ni = torch.arange(nb)
    maskids = (torch.arange(start=0,end=n)).repeat(n).repeat(nt)
    targetids = torch.arange(start=0,end=n).repeat_interleave(n).repeat(nt)
    tokenids = torch.tensor(alphabet_num).repeat_interleave(n*(n-1)).to(device=seq_tok.device)

    mm = maskids != targetids
    maskids = maskids[mm]
    targetids = targetids[mm]
    seq_batch[ni,maskids] = mask_id
    seq_batch[ni,targetids] = tokenids
    return seq_batch, maskids,targetids,tokenids

def compute_conditional_prob_matrix(model,seqs,maskids,targetids,tokenids,maxresidues,device,ncategories):
    """
    This runs a series a sequences (seqs) through the neural network (model),
    and saves the selected predicted probabilities according to maskids, targetids into a conditional probability matrix (M) of shape (l,l,ncategories,ncategories)
    M is ordered such that the first dimension corresponds to the mask residue, while the second dimension correponds to the target residue.
    The third dimension gives the tokenids of the target, while the forth dimension corresponds to the actual masked probabilities.
    Hence:
        M[2,5,3,:] corresponds to the masked probabilities from a sequence,
        where the 2nd residue was masked and the 5th residue was altered to be the 4th letter in the alphabet (counting starts at zero)
    """

    nb,l = seqs.shape
    assert maxresidues >= l,  "The protein is longer than the maximum number of residues allowed at once."
    seq_pr_iteration = int(np.floor(maxresidues/l))
    niters = int(np.ceil(nb/seq_pr_iteration)) # Compute the number of iterations we will have to split the job into in order to calculate it.
    M = torch.zeros((l,l,ncategories,ncategories),dtype=torch.float32,device=device)

    for i in range(niters):
        i0 = i*seq_pr_iteration
        i1 = (i+1)*seq_pr_iteration
        maskid = maskids[i0:i1]
        targetid = targetids[i0:i1]
        tokenid = tokenids[i0:i1]
        seq_batch = seqs[i0:i1,:]
        with torch.no_grad():
            prob = model(seq_batch)
            tmp = torch.arange(maskid.shape[0], device=device)
            pselect = prob[tmp,maskid,:]
            M[maskid,targetid,tokenid,:] = pselect
    return M

def compute_kl_divergence_matrix(q,p):
    """
    q is a (l,l,nA) shaped matrix, with masked pair probabilities
    p is a (l,l,nA,nA) shaped matrix,
    """
    l, _, nA = q.shape
    M = torch.zeros((l, l), dtype=torch.float32, device=q.device)
    for i in range(l):
        for j in range(l):
            if i==j:
                continue
            for k in range(nA):
                tmp =torch.sum(p[i, j, k, :] * torch.log(p[i, j, k, :] / (q[i, j, :] * q[j, i, k])))
                if torch.isnan(tmp) or torch.isinf(tmp):
                    print("stop")
                M[i,j] += tmp
    return M


if __name__ == '__main__':
    seq = torch.tensor([0, 1, 1])
    seq_batch, ii,jj,tok_i = mask_scheme_cov(seq,test_toks=[0,1])
