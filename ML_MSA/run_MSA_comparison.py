import glob
import torch.nn.functional as F

import torch
import esm
import time
import numpy as np

from ML_MSA.MSA_generators import mask_scheme1, MSA_once
from ML_MSA.MSA_reader import read_a2m_gz_file
from ML_MSA.MSA_utils import msa2weight, msa2pssm, msa2cov, setup_protein_comparison
from src.dataloader_pnet import parse_pnet
from src.network_transformer import tr2DistSmall
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') #TkAgg

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
# model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
aa_list = alphabet.all_toks
aa_eff = aa_list[4:-4]
batch_converter = alphabet.get_batch_converter()
# batch_converter = BatchConverter(alphabet)

# seqs = ["MYMKK", "MYYKK"]
# strs, tokens = batch_converter(seqs)

# data = [("protein1", "MYLYQKIKN"), ("protein2", "MNAKYD")]
# batch_labels, batch_strs, batch_tokens = batch_converter(data)

#Load sequence
folder = 'F:/Globus/raw/'
search_command = folder + "*.a2m.gz"
a2mfiles = [f for f in glob.glob(search_command)]

#Find the indices we want to try:
folder = 'F:/Output/'
search_command = folder + "*.npz"
npzfiles = [f for f in glob.glob(search_command)]

n_pnet = 20000

pnet_file = './../data/training_30.pnet'
tt = time.time()
args = parse_pnet(pnet_file, AA_list=aa_list, unk_idx=alphabet.unk_idx, use_dssp=False, use_pssm=False, use_mask=False, use_entropy=False)
seqs = args['seq']
seqs_len = args['seq_len']
r1 = args['r1']
r2 = args['r2']
r3 = args['r3']
n_samples = len(seqs_len)
print("Read the pnet_file, took {:2.2f}s, contains {:} samples".format(time.time() - tt, n_samples))
seqs_list, seqs_list_org_id, lookup = setup_protein_comparison(seqs, seqs_len)

for ii,npzfile in enumerate(npzfiles):
    if ii<10:
        continue
    file_index = int(npzfile.split("_")[-1].split(".")[0])
    # file_index = 4068
    # filename = 'F:/Globus/raw_subset/3U9P_d3u9pm2.a2m.gz'
    filename = a2mfiles[file_index] #103 123 125 130 140

    msa_pnet = read_a2m_gz_file(filename,aa_list,alphabet.unk_idx)
    seq = msa_pnet[-1,:]
    l = seq.shape[-1]
    print("{:}, ID={:}, Data sample has length {:} and {:} MSAs in pnet".format(ii,file_index,msa_pnet.shape[-1],msa_pnet.shape[0]))
    # seq_tok = 'GSMANKPMQPITSTANKIVWSDPTRLSTTFSASLLRQRVKVGIAELNNVSGQYVSVYKRPAPKPEGGADAGVIMPNENQSIRTVISGSAENLATLKAEWETHKRNVDTLFASGNAGLGFLDPTAAIVSSDTTA'
    # data = [("protein1", seq_tok), ]
    # batch_labels, batch_strs, seq = batch_converter(data)
    # seq = seq[:,1:-1]

    # Pnet stuff
    indices = np.random.permutation(msa_pnet.shape[0])
    idx = indices[:n_pnet]
    msa_pnet = msa_pnet[idx,:]
    msa_pnet = msa_pnet - 4

    #Lets convert all unk to X
    tmp = msa_pnet == -1
    msa_pnet[tmp] = 20
    t1 = time.time()
    msa1hot_pnet = torch.nn.functional.one_hot(msa_pnet, num_classes=len(aa_eff))
    t2 = time.time()
    msa1hot_pnet = msa1hot_pnet.to(device=device, dtype=torch.float16)

    w_pnet, Nf_pnet = msa2weight(msa1hot_pnet)
    pssm_pnet, entropy_pnet = msa2pssm(msa1hot_pnet, w_pnet)
    _, contact_pnet = msa2cov(msa1hot_pnet, w_pnet, penalty=4.5)
    msa1hot_pnet = None
    w_pnet = None
    t3 = time.time()
    print("pnet-MSA Nf = {:2.2f} ".format(Nf_pnet))
    # out = F.kl_div(pssm_pnet, pssm_pnet)
    # out2 = F.kl_div(pssm_pnet, pssm_pnet,log_target=True)




    # seq = "MYMKKLKFITLTLAIIITLPMLQSCLDDNDHQSRSLVISTINQISEDSKEFYFTLDNGKTMFPSNSQAWGGEKFENGQRAFVIFNELEQPVNGYDYNIQVRDITKVLTKEIVTMDDEENTEEKIGDDKINATYMWISKDKKYLTIEFQYYSTHSEDKKHFLNLVINNKEADSAAENEDNTDDEYINLEFRHNSERDSPDHLGEGYVSFKLDKIEEQIEGKKGLNIRVRTLYDGIKNYKVQFP"
    # data = [("sequence",seq),]
    # _,_,seq_tok = batch_converter(data)
    ite = 0
    max_iter = 10
    p_mask_max = 0.5
    while ite < max_iter:
        n = l * 10
        nb = 100

        p_mask = p_mask_max*(ite+1)/(max_iter+1)
        # p_mask = 0.1+p_mask_max*(ite+1)/(max_iter+1)
        t0 = time.time()
        seqs,masks = mask_scheme1(seq,n=n,p_mask=p_mask)
        seqs = seqs.to(device)
        # seq_batch = seqs.to(device)
        MSA = torch.empty_like(seqs)
        tt0 = time.time()
        model.to(device)
        tt1 = time.time()
        print("model to gpu time {:2.2f}, p_mask={:2.2f}".format(tt1-tt0,p_mask))
        for i in range(np.int(np.ceil(n/nb))):
            i0 = i*nb
            i1 = (i+1)*nb
            seq_batch = seqs[i0:i1,:]
            seq_batch = seq_batch.to(device)
            MSA[i0:i1,:] = MSA_once(seq_batch,model,deterministic=False)
        model.to('cpu')
        MSA = MSA[:, 1:-1]
        MSA = MSA
        MSA = MSA - 4

        if ite == 0:
            MSAS = MSA
        else:
            MSAS = torch.cat((MSAS,MSA),dim=0)
            MSA = None
        t1 = time.time()
        MSA1hot = torch.nn.functional.one_hot(MSAS, num_classes=len(aa_eff))
        MSA1hot = MSA1hot.to(device=device, dtype=torch.float16)
        t2 = time.time()
        w, Nf = msa2weight(MSA1hot)
        print("Iterative MSA N={:} Nf ={:2.2f} ".format(MSAS.shape[0],Nf))
        if Nf > max(Nf_pnet,100):
            break
        ite += 1

    nmsas = MSAS.shape[0]
    model.to("cpu")
    MSAS = None
    MSA = None
    pssm, entropy = msa2pssm(MSA1hot, w)
    _, contact = msa2cov(MSA1hot, w, penalty=4.5)
    contact = contact.cpu()
    MSA1hot = MSA1hot.cpu()
    t3 = time.time()

    print("{:2.2f} {:2.2f} {:2.2f}".format(t1-t0,t2-t1,t3-t2))
    print("ML-MSA N = {:}, Nf = {:2.2f} ".format(nmsas, Nf))



    seq_len = seq.shape[-1]
    try:
        seq_idx = lookup[seq_len]
    except:
        pass
    seqs_i = seqs_list[seq_idx]
    res = np.mean(seq.numpy() == seqs_i, axis=1) == 1
    if np.sum(res) == 1:
        org_id = (seqs_list_org_id[seq_idx][res]).squeeze()
        coords = r1[org_id].T
        D = tr2DistSmall(torch.tensor(coords[None,:,:]))
    else:
        print("sample not found in pnet, will not be able to show distogram")
        D = None


    fig = plt.figure(num=1, figsize=[15, 10])
    plt.clf()

    plt.subplot(1,3,1)
    plt.imshow(contact.cpu())
    plt.title("ML MSA, N {:},  Nf {:2.2f}".format(nmsas,Nf))
    plt.subplot(1,3,2)
    plt.imshow(contact_pnet.cpu())
    plt.title("pnet MSA, Nf {:2.2f}".format(Nf_pnet))
    if D is not None:
        plt.subplot(1,3,3)
        plt.imshow(D.squeeze())
        plt.title("Distogram")
    plt.pause(1)
    save = "./../figures/{}.png".format(file_index)
    fig.savefig(save)

    plt.clf()

    plt.subplot(2, 1, 1)
    plt.imshow(pssm.cpu().float().transpose(0,1))
    plt.title("ML PSSM, N {:},  Nf {:2.2f}".format(nmsas, Nf))
    plt.subplot(2, 1, 2)
    plt.imshow(pssm_pnet.cpu().float().transpose(0,1))
    plt.title("pnet, Nf {:2.2f}".format(Nf_pnet))
    plt.pause(1)
    save = "./../figures/PSSM_{}.png".format(file_index)
    fig.savefig(save)

    print("Memory allocated = {:2.4f} GB, Max memory {:2.4f} GB ".format(torch.cuda.memory_allocated(device)/1024/1024/1024,torch.cuda.max_memory_allocated(device)/1024/1024/1024))
    # torch.save()

input("Press Enter to continue...")
print("done")