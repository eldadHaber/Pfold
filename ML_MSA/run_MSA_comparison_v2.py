import glob
import torch.nn.functional as F

import torch
import esm
import time
import numpy as np

from ML_MSA.MSA_generators import mask_scheme1, MSA_once, mask_scheme_cov, mask_scheme_cov_ref
from ML_MSA.MSA_reader import read_a2m_gz_file
from ML_MSA.MSA_utils import msa2weight, msa2pssm, msa2cov, setup_protein_comparison, AA_converter
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

AA = AA_converter(aa_list,alphabet.unk_idx)
test_toks = AA(aa_eff)
n_test_toks = len(test_toks)
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

# pnet_file = './../data/training_30.pnet'
# tt = time.time()
# args = parse_pnet(pnet_file, AA_list=aa_list, unk_idx=alphabet.unk_idx, use_dssp=False, use_pssm=False, use_mask=False, use_entropy=False)
# seqs = args['seq']
# seqs_len = args['seq_len']
# r1 = args['r1']
# r2 = args['r2']
# r3 = args['r3']
# n_samples = len(seqs_len)
# print("Read the pnet_file, took {:2.2f}s, contains {:} samples".format(time.time() - tt, n_samples))
# seqs_list, seqs_list_org_id, lookup = setup_protein_comparison(seqs, seqs_len)

for ii,npzfile in enumerate(npzfiles):
    # if ii<10:
    #     continue
    file_index = int(npzfile.split("_")[-1].split(".")[0])
    file_index = 10078
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

    nb = 100
    seqs_ref, ii_ref, jj_ref = mask_scheme_cov_ref(seq)
    seqs_ref = seqs_ref.to(device)
    ii_ref = ii_ref.to(device)
    jj_ref = jj_ref.to(device)
    model.to(device)
    n = seqs_ref.shape[0]
    cov_ref = torch.zeros((l,l,n_test_toks),device=device)
    tt0 = time.time()
    maxiter = np.int(np.ceil(n/nb))

    # for i in range(maxiter):
    #     tt1 = time.time()
    #     i0 = i*nb
    #     i1 = (i+1)*nb
    #     seq_ref_batch = seqs_ref[i0:i1,:]
    #     iii = ii_ref[i0:i1]
    #     jjj = jj_ref[i0:i1]
    #     # seq_batch = seq_batch.to(device)
    #     tt2 = time.time()
    #     with torch.no_grad():
    #         results = model(seq_ref_batch)
    #         tt3 = time.time()
    #         pred = results['logits']
    #         a=torch.arange(iii.shape[0],device=device)
    #         pred = pred[a,iii,:]
    #         pred = pred[:,4:-4]
    #         prob = torch.softmax(pred, dim=1)
    #         peff = prob
    #         cov_ref[iii-1,jjj-1,:] = peff
    #         tt4 = time.time()
    #     print("{:}/{:} times {:2.2f},{:2.2f},{:2.2f},{:2.2f}s, ETA {:2.2f}h".format(i,maxiter,tt2-tt1,tt3-tt2,tt4-tt3,tt4-tt0,(tt4-tt0)/(i+1)*(maxiter-(i+1))/3600))
    #
    # save = "./../figures/cov_ref_{}.pt".format(file_index)
    # torch.save(cov_ref, save)

    # seq = "MYMKKLKFITLTLAIIITLPMLQSCLDDNDHQSRSLVISTINQISEDSKEFYFTLDNGKTMFPSNSQAWGGEKFENGQRAFVIFNELEQPVNGYDYNIQVRDITKVLTKEIVTMDDEENTEEKIGDDKINATYMWISKDKKYLTIEFQYYSTHSEDKKHFLNLVINNKEADSAAENEDNTDDEYINLEFRHNSERDSPDHLGEGYVSFKLDKIEEQIEGKKGLNIRVRTLYDGIKNYKVQFP"
    # data = [("sequence",seq),]
    # _,_,seq_tok = batch_converter(data)
    nb = 100
    seqs, ii, jj, toks = mask_scheme_cov(seq, test_toks=test_toks)
    seqs = seqs.to(device)
    ii = ii.to(device)
    jj = jj.to(device)
    toks = toks.to(device)
    model.to(device)
    n = seqs.shape[0]
    cov = torch.ones((l,l,n_test_toks,n_test_toks),device=device)
    prob_ref = 0
    tt0 = time.time()
    maxiter = np.int(np.ceil(n/nb))
    for i in range(maxiter):
        tt1 = time.time()
        i0 = i*nb
        i1 = (i+1)*nb
        seq_batch = seqs[i0:i1,:]
        iii = ii[i0:i1]
        jjj = jj[i0:i1]
        prob_ref = cov_ref[iii-1,jjj-1,:]
        tok_i = toks[i0:i1]
        # seq_batch = seq_batch.to(device)
        tt2 = time.time()
        with torch.no_grad():
            results = model(seq_batch)
            tt3 = time.time()
            pred = results['logits']
            a=torch.arange(iii.shape[0],device=device)
            pred = pred[a,iii,:]
            pred = pred[:,4:-4]
            prob = torch.softmax(pred, dim=1)
            peff = prob - prob_ref
            cov[iii-1,jjj-1,tok_i-4,:] = peff
            tt4 = time.time()
        print("{:}/{:} times {:2.2f},{:2.2f},{:2.2f},{:2.2f}s, ETA {:2.2f}h".format(i,maxiter,tt2-tt1,tt3-tt2,tt4-tt3,tt4-tt0,(tt4-tt0)/(i+1)*(maxiter-(i+1))/3600))

    #So we end up with cov of shape (l,l,nA,nA), where the first dimension is the masked residue, the second dimension is the one it is compared to, which is set the value in the third dimension, giving the probability in the 4th dimension


    save = "./../figures/cov_{}.pt".format(file_index)
    torch.save(cov, save)

    import matplotlib
    import matplotlib.pyplot as plt
    nA = cov.shape[-1]
    l = cov.shape[0]
    fig = plt.figure(figsize=(15, 10))
    for i in range(nA):
        for j in range(nA):
            plt.clf()
            plt.imshow((cov[:,:,i,j].cpu()))
            plt.title("Given amino acid i={:}, the probability for amino acid j={:} for all residue combinations.".format(i,j))
            plt.xlabel("masked residue")
            plt.ylabel("paired residue")
            plt.colorbar()
            save = "./../figures/cov_{:}_{:}_{:}".format(file_index,i,j)
            plt.savefig("{:}.png".format(save))
            # plt.pause(1)



    for i in range(l):
        for j in range(l):
            plt.clf()
            plt.imshow((cov[i,j,:,:].cpu()))
            plt.title("residue i={:},j={:}".format(i,j))
            plt.title("Residue i={:} is masked, and residue j={:} is set to the following amino acid.".format(i,j))
            plt.xlabel("amino acid of residue j")
            plt.ylabel("predicted amino acid of i")
            plt.colorbar()
            save = "./../figures/residue_cov_{:}_{:}_{:}".format(file_index,i,j)
            plt.savefig("{:}.png".format(save))
            # plt.pause(1)




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