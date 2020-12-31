import torch
import esm
import time
import numpy as np
import random
# Load 34 layer model
# model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()


def mask_scheme1(seq_tok,n=10,p_mask=0.2,mask_id=32):
    l = seq_tok.shape[-1]
    rng = np.random.default_rng()
    pm = rng.random((n,l))
    seq_batch = seq_tok.repeat(n,1)
    idx = torch.from_numpy(pm < p_mask)
    seq_batch[idx] = mask_id
    seq_batch[:,0] = 0 #Start token
    seq_batch[:,-1] = 2 #End token
    mask = seq_batch == mask_id
    return seq_batch, mask

def bincount2d(arr, bins=None):
    if bins is None:
        bins = np.max(arr) + 1
    count = np.zeros(shape=[len(arr), bins], dtype=np.int64)
    indexing = (np.ones_like(arr).T * np.arange(len(arr))).T
    np.add.at(count, (indexing, arr), 1)

    return count

def msa_to_pssm(MSA):
    #Assume MSA has shape (n,l) and is a numpy array
    counts = bincount2d(MSA.T,bins=33)
    n = MSA.shape[0]
    pssm = counts.T/n
    return pssm




device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
model.to(device)
batch_converter = alphabet.get_batch_converter()

# Prepare data (two protein sequences)
# data = [("protein1", "MYLYQKIKN"), ("protein2", "MNAKYD")]
# batch_labels, batch_strs, batch_tokens = batch_converter(data)

seq = "MYMKKLKFITLTLAIIITLPMLQSCLDDNDHQSRSLVISTINQISEDSKEFYFTLDNGKTMFPSNSQAWGGEKFENGQRAFVIFNELEQPVNGYDYNIQVRDITKVLTKEIVTMDDEENTEEKIGDDKINATYMWISKDKKYLTIEFQYYSTHSEDKKHFLNLVINNKEADSAAENEDNTDDEYINLEFRHNSERDSPDHLGEGYVSFKLDKIEEQIEGKKGLNIRVRTLYDGIKNYKVQFP"
# seq = "MYMKKLKFITLT"
data = [("sequence",seq),]
_,_,seq_tok = batch_converter(data)
n = 100

seqs,masks = mask_scheme1(seq_tok,n=n)
seq_batch = seqs.to(device)
mask = masks.numpy()

aa = alphabet.all_toks
# model.model_version
# Extract per-residue embeddings (on CPU)
ite = 0
batch_idx = torch.arange(n)
MSA_iter = seq_batch.cpu().numpy()
with torch.no_grad():
    while True:
        t0 = time.time()
        results = model(seq_batch)
        pred = results['logits']
        prob = torch.softmax(pred,dim=2)
        p = prob.cpu().numpy()
        t1 = time.time()
        if ite == 0:
            t1s = time.time()
            MSA_once = seq_batch.cpu().numpy()
            aa_select = np.argmax(p,axis=2)
            MSA_once[mask] = aa_select[mask]
            t2s = time.time()
            print("Saving the MSA_once took {:2.2f}s".format(t2s-t1s))

        t2 = time.time()
        idx = np.argmax(p,axis=2)
        val = np.max(p,axis=2)
        nm = np.sum(mask,axis=1)
        nmtot = np.sum(mask)
        t3 = time.time()

        if np.sum(nm==0) > 0:
            MSA_iter[batch_idx[nm==0],:] = seq_batch[nm==0,:].cpu().numpy()
        t4 = time.time()

        if nmtot == 0:
            break
        val = val[nm > 0,:]
        idx = idx[nm > 0,:]
        mask = mask[nm > 0,:]
        t45 = time.time()
        seq_batch = seq_batch[nm > 0, :]
        batch_idx = batch_idx[nm > 0]
        t5 = time.time()

        nremain = seq_batch.shape[0]

        val_m = val
        val_m[mask == False] = 0


        next_aa = np.argmax(val_m,axis=1)
        t6 = time.time()

        nn = np.arange(nremain)
        mask[nn,next_aa] = False
        seq_batch[nn, next_aa] = torch.from_numpy(idx[nn, next_aa]).to(device)
        t7 = time.time()
        # print("{:2.2f}s".format(t5-t45))
        print("{:} sequences took {:2.2f}s, {:2.2f}s {:2.2f}s {:2.2f}s {:2.2f}s {:2.2f}s {:2.2f}s".format(nremain,t1-t0,t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6))
        # print("now what?")
        ite += 1

    MSA_iter = MSA_iter[:,1:-1]
    MSA_once = MSA_once[:,1:-1]
    pssm_iter = msa_to_pssm(MSA_iter)
    pssm_once = msa_to_pssm(MSA_once)

    print("done")