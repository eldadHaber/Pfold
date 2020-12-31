import torch
import esm
import time
import numpy as np
import random
# Load 34 layer model
# model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()




device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
# model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
aa = alphabet.all_toks
model.to(device)
batch_converter = alphabet.get_batch_converter()

# Prepare data (two protein sequences)
data = [("protein1", "MYLYQKIKN"), ("protein2", "MNAKYD")]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# msas = read_a2m_gz_file(a2mfile, verbose=True)

protein = msas[-1, :]
n = msas.shape[0]
indices = np.random.permutation(n)
pssms = []
f2d_dcas = []
for j in range(max(min(n_repeats, n // max_samples), 1)):
    idx = indices[j * max_samples:min((j + 1) * max_samples, n)]
    msa = msas[idx, :]
    x_sp = sparse_one_hot_encoding(msa)

    nsub = min(nsub_size, n)
    w = ANN_sparse(x_sp[0:nsub], x_sp, k=100, eff=300, cutoff=True)
    mask = w == np.min(w)
    wm = np.sum(w[mask])
    if wm > 0.01 * np.sum(
            w):  # If these account for more than 1% of the total weight, we scale them down to almost 1%.
        scaling = wm / np.sum(w) / 0.01
    else:
        scaling = 1
    w[mask] /= scaling

    wn = w / np.sum(w)
    msa_1hot = np.eye(21, dtype=np.float32)[msa]
    t5 = time.time()
    t.ann_time.add_time(t5 - t4)
    pssms.append(msa2pssm(msa_1hot, w))
    f2d_dcas.append(dca(msa_1hot, w))
    t6 = time.time()
    t.dca_time.add_time(t6 - t5)
f2d_dca = np.mean(np.asarray(f2d_dcas), axis=0)
pssm = np.mean(np.asarray(pssms), axis=0)

# SAVE FILE HERE
if IDs is None:
    fullfileout = "{:}ID_{:}".format(outputfolder, i)
else:
    fullfileout = "{:}ID_{:}".format(outputfolder, IDs[i])
np.savez(fullfileout, protein=protein, pssm=pssm, dca=f2d_dca, r1=r1[org_id].T, r2=r2[org_id].T, r3=r3[org_id].T)

elif np.sum(res) == 0:
    c.MSAs_no_match += 1
else:
    c.MSAs_multi_match += 1
if (i + 1) % report_freq == 0:
    print(
        "Compared {:} proteins. Matches: {:}, MSA not in pnet: {:}, MSAs in pnet more than once {:}, excluded: {:}, Avr Time(read): {:2.2f}, Avr Time(lookup): {:2.2f}, Avr Time(ANN): {:2.2f}, Avr Time(DCA): {:2.2f}, Total Time(read): {:2.2f}, Total Time(lookup): {:2.2f}, Total Time(ANN): {:2.2f}, Total Time(DCA): {:2.2f} Time(total): {:2.2f}".format(
            i + 1, c.match, c.MSAs_no_match, c.MSAs_multi_match, c.excluded, t.read_time(), t.lookup_time(),
            t.ann_time(), t.dca_time(), t.read_time(total=True), t.lookup_time(total=True), t.ann_time(total=True),
            t.dca_time(total=True), time.time() - t.t0))

# finish iterating through dataset
print(
    "Compared {:} proteins. Matches: {:}, MSA not in pnet: {:}, MSAs in pnet more than once {:}, excluded: {:}, Avr Time(read): {:2.2f}, Avr Time(lookup): {:2.2f}, Avr Time(ANN): {:2.2f}, Avr Time(DCA): {:2.2f}, Total Time(read): {:2.2f}, Total Time(lookup): {:2.2f}, Total Time(ANN): {:2.2f}, Total Time(DCA): {:2.2f} Time(total): {:2.2f}".format(
        i + 1, c.match, c.MSAs_no_match, c.MSAs_multi_match, c.excluded, t.read_time(), t.lookup_time(),
        t.ann_time(), t.dca_time(), t.read_time(total=True), t.lookup_time(total=True), t.ann_time(total=True),
        t.dca_time(total=True), time.time() - t.t0))

seq = "MYMKKLKFITLTLAIIITLPMLQSCLDDNDHQSRSLVISTINQISEDSKEFYFTLDNGKTMFPSNSQAWGGEKFENGQRAFVIFNELEQPVNGYDYNIQVRDITKVLTKEIVTMDDEENTEEKIGDDKINATYMWISKDKKYLTIEFQYYSTHSEDKKHFLNLVINNKEADSAAENEDNTDDEYINLEFRHNSERDSPDHLGEGYVSFKLDKIEEQIEGKKGLNIRVRTLYDGIKNYKVQFP"
# seq = "MYMKKLKFITLT"
data = [("sequence",seq),]
_,_,seq_tok = batch_converter(data)
n = 10000
nb = 100

t0 = time.time()
seqs,masks = mask_scheme1(seq_tok,n=n,p_mask=0.2)
seqs = seqs.to(device)
# seq_batch = seqs.to(device)
MSA = torch.empty_like(seqs)
for i in range(np.int(np.ceil(n/nb))):
    i0 = i*nb
    i1 = (i+1)*nb
    seq_batch = seqs[i0:i1,:]
    seq_batch = seq_batch.to(device)
    MSA[i0:i1,:] = MSA_once(seq_batch)
MSA = MSA[:, 1:-1]
MSA = MSA
aa_eff = aa[4:-6]
MSA = MSA - 4

t1 = time.time()
MSA1hot = torch.nn.functional.one_hot(MSA, num_classes=len(aa_eff)).float()
t2 = time.time()
w, Nf = Neff_MSA(MSA1hot.cuda())
pssm, entropy = msa2pssm(MSA1hot, w)
inv_cov, contact = dca(MSA1hot, w, penalty=4.5)
t3 = time.time()

print("{:2.2f} {:2.2f} {:2.2f}".format(t1-t0,t2-t1,t3-t2))
print("Nf = {:2.2f} ".format(Nf))

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') #TkAgg

plt.imshow(contact.cpu())
plt.pause(1)
print("done")