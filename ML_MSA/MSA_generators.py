import torch
import time
import numpy as np

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


def mask_scheme_cov_ref(seq_tok,mask_id=32,prepend_idx=0,append_idx=2):
    """This will o nly work with the append prepend idx used"""
    n = seq_tok.shape[-1]
    n_added_tokens = 0
    if prepend_idx >= 0:
        n_added_tokens += 1
    if append_idx >= 0:
        n_added_tokens += 1
    l = n + n_added_tokens
    nc = n*(n-1)
    nb = nc
    seq_rep = seq_tok.repeat(nb,1)
    seq_batch = torch.empty(nb,l,dtype=torch.int64,device=seq_tok.device)
    seq_batch[:,1:-1] = seq_rep
    # seq_batch[:,:] = seq_rep
    ni = torch.arange(nb)
    ii = (torch.arange(start=1,end=n+1)).repeat(n)
    jj = torch.arange(start=1,end=n+1).repeat_interleave(n)

    mm = ii != jj
    ii = ii[mm]
    jj = jj[mm]
    seq_batch[ni,ii] = mask_id
    seq_batch[ni,jj] = mask_id
    seq_batch[:,0] = 0 #Start token
    seq_batch[:,-1] = 2 #End token
    # mask = seq_batch == mask_id
    return seq_batch, ii,jj


if __name__ == '__main__':
    seq = torch.tensor([0, 1, 1])
    seq_batch, ii,jj,tok_i = mask_scheme_cov(seq,test_toks=[0,1])
