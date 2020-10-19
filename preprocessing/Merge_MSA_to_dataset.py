import glob
from numpy.linalg import svd, eig, eigh
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import matplotlib
import copy
matplotlib.use('TkAgg') #TkAgg


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    b = np.swapaxes(a,-1,-2)
    return np.allclose(a, b, rtol=rtol, atol=atol)

def extract_1d_features(A,k=130,debug=False,safety=True):

    if safety:
        assert check_symmetric(A)

    t0 = time.time()
    w, v = eigh(A)
    idxs = np.argsort(np.abs(w),axis=-1)[:, ::-1]
    e = np.ones(len(idxs[0]),dtype=np.int)
    idxs_e = idxs[:,None,:] * e[None,:,None]
    idxs_e[:,:,:] = idxs_e
    t1 = time.time()

    I = np.arange(v.shape[0])[:, None]
    V = v[np.arange(v.shape[0])[:, None], :, idxs].swapaxes(1, 2)
    # V = np.empty_like(A)
    # V[I, :, np.arange(len(idxs[0]))] = v[I, :, idxs]

    # for i in range(A.shape[0]):
    #     V[i,:,np.arange(len(idxs[0]))] = v[i,:,idxs[i,:]]
    #
    # V2 = np.empty_like(A)



    # V =
    t2 = time.time()
    if debug:
        # W = np.diag(w[idx])
        # A_pc = V[:, :k] @ W[:k, :k] @ V[:, :k].T
        W = np.empty((A.shape[0],k,k),dtype=A.dtype)

        for i in range(A.shape[0]):
            W[i,:,:] = np.diag(w[i,idxs[i,:k]])
        A_pc = V[:, :, :k] @ W @ np.swapaxes(V[:, :, :k],axis1=-1, axis2=-2)

        A_dif = A - A_pc
        A_dif_percent = np.linalg.norm(A_dif,axis=(1,2)) / np.linalg.norm(A,axis=(1,2))*100
        print("1D feature extraction error= {:2.2f}%, max={:2.2f}%, time taken: {:2.2f}s".format(np.mean(A_dif_percent),np.max(A_dif_percent),t1-t0))

    permutation = [0, 4, 1, 3, 2]
    idxss = np.empty_like(idxs[0,:])
    idxss[idxs[0,:]] = np.arange(len(idxs[0,:]))
    t3 = time.time()
    print("Times {:2.2f}, {:2.2f}, {:2.2f}".format(t1-t0,t2-t1,t3-t2))
    return V[:,:,:k]

if __name__ == "__main__":
    from pathlib import Path
    MSA_folder = "./../data/MSA/"
    main_folder = "./../data/clean_pnet_test/"
    output_folder = "./../data/train/"
    report_iter = 500

    # Read all of main_folder in
    search_command = main_folder + "*.npz"
    main_files = [f for f in glob.glob(search_command)]

    seqs = []
    seqs_len = np.empty(len(main_files),dtype=np.int)
    pssms = []
    entropys = []
    r1s = []
    r2s = []
    r3s = []
    # masks = []
    t0 = time.time()
    for i, main_file in enumerate(main_files):
        data = np.load(main_file)
        seqs.append(data['seq'])
        seqs_len[i] = len(data['seq'])
        pssms.append(data['pssm'])
        entropys.append(data['entropy'])
        r1s.append(data['r1'])
        r2s.append(data['r2'])
        r3s.append(data['r3'])
        if (i+1) % report_iter == 0:
            print("{:}/{:} Time: {:2.0f}".format(i+1,len(main_files),time.time()-t0))

    seqs_len_unique, counts = np.unique(seqs_len, return_counts=True)
    a = np.digitize(seqs_len, bins=seqs_len_unique, right=True)
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

    counter = np.zeros_like(counts)
    for i, (seq, seq_len) in enumerate(zip(seqs, seqs_len)):
        seq_idx = lookup[seq_len]
        counter_idx = counter[seq_idx]
        seqs_list[seq_idx][counter_idx, :] = seq
        seqs_list_org_id[seq_idx][counter_idx] = i
        counter[seq_idx] += 1


    # Read MSA_folder in one at a time and match them up against main
    search_command = MSA_folder + "*.npz"
    MSA_files = [f for f in glob.glob(search_command)]
    for i, MSA_file in enumerate(MSA_files):
        data = np.load(MSA_file, allow_pickle=True)
        dca = data['dca']
        protein = data['protein']
        seq_len = len(protein)
        try:
            seq_idx = lookup[seq_len]
        except:
            print("Protein not in database")
            # continue
        seqs_i = seqs_list[seq_idx]
        # res = np.mean(protein == seqs_i, axis=1) == 1

        # Ensure dca is symmetric, which it was supposed to be
        dca = 0.5 * (dca + np.swapaxes(dca,0,1))


        cov = dca[:,:,:-1]
        contact = dca[:,:,-1]

        k = 10
        cov = np.swapaxes(cov,0,2)

        contact_features = extract_1d_features(contact[None,:,:], k=k, debug=True, safety=True)
        cov_features = extract_1d_features(cov, k=k, debug=True, safety=True)

        t0 = time.time()
        u,s,v = svd(cov,hermitian=True,full_matrices=False)
        t1 = time.time()
        print("svd took : {:2.2f}".format(t1-t0))

        t0 = time.time()
        w,vv = eigh(cov)
        t1 = time.time()
        print("eig took : {:2.2f}".format(t1-t0))


        cc = cov[1,:,:]
        W = np.diag(w)
        V = vv
        cov_eig = V @ W @ V.T


        cov_svd = np.empty_like(cov)
        for i in range(u.shape[0]):
            cov_svd[i,:] = np.dot(u[i,:, :k], np.dot(np.diag(s[i,:k]), v[i,:k, :]))

        dif = cov - cov_svd
        dif_percent = np.linalg.norm(dif,axis=(1,2)) / np.linalg.norm(cov,axis=(1,2))*100
        print("Cov error: {:2.2f}%, max={:}%".format(np.mean(dif_percent),np.max(dif_percent)))

        t0 = time.time()
        u, s, v = svd(contact, full_matrices=False, hermitian=True)
        t1 = time.time()

        print("svd took : {:2.2f}".format(t1-t0))
        contact_svd = np.dot(u[:, :k], np.dot(np.diag(s[:k]), v[:k, :]))
        t0 = time.time()
        w,vv = eig(contact[:,:])
        t1 = time.time()
        print("eig took : {:2.2f}".format(t1-t0))
        idx = np.argsort(np.abs(w))[::-1]
        W = np.diag(w[idx])
        V = vv[:,idx]
        # contact_eig = V @ W @ V.T
        contact_eig = V[:, :k] @ W[:k, :k] @ V[:, :k].T

        plt.figure(figsize = (15, 10))
        dif = contact-contact_svd
        dif_percent = np.linalg.norm(dif)/np.linalg.norm(contact)*100
        print("Contact error: {:2.2f}%".format(dif_percent))
        plt.subplot(1,4,1)
        plt.imshow(contact)
        plt.title("org")
        plt.colorbar()
        plt.subplot(1,4,2)
        plt.imshow(contact_svd)
        plt.title("svd={:}".format(k))
        plt.colorbar()
        plt.subplot(1,4,3)
        plt.imshow(np.abs(contact-contact_svd))
        plt.colorbar()
        plt.subplot(1,4,4)
        plt.imshow(contact_eig)
        plt.colorbar()
        plt.title("eig={:}".format(k))
        plt.pause(1)

        j = 1
        plt.figure()
        plt.subplot(1,4,1)
        plt.imshow(cov[j,:,:])
        plt.colorbar()
        plt.title("org")
        plt.subplot(1,4,2)
        plt.imshow(cov_svd[j,:,:])
        plt.title("svd={:}".format(k))
        plt.colorbar()
        plt.subplot(1,4,3)
        plt.imshow(np.abs(cov[j,:,:]-cov_svd[j,:,:]))
        plt.colorbar()
        plt.subplot(1,4,4)
        plt.imshow(cov_eig)
        plt.colorbar()
        plt.title("eig={:}".format(k))
        plt.pause(1)


        j = 17
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(cov[j,:,:])
        plt.title("org")
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.imshow(cov_svd[j,:,:])
        plt.title("svd={:}".format(k))
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.imshow(np.abs(cov[j,:,:]-cov_svd[j,:,:]))
        plt.colorbar()
        plt.pause(1)


        j = 113
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(cov[j,:,:])
        plt.title("org")
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.imshow(cov_svd[j,:,:])
        plt.title("svd={:}".format(k))
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.imshow(np.abs(cov[j,:,:]-cov_svd[j,:,:]))
        plt.colorbar()
        plt.pause(1)



        j = 201
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(cov[j,:,:])
        plt.title("org")
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.imshow(cov_svd[j,:,:])
        plt.title("svd={:}".format(k))
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.imshow(np.abs(cov[j,:,:]-cov_svd[j,:,:]))
        plt.colorbar()
        plt.pause(1)


        j = 202
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(cov[j,:,:])
        plt.title("org")
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.imshow(cov_svd[j,:,:])
        plt.title("svd={:}".format(k))
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.imshow(np.abs(cov[j,:,:]-cov_svd[j,:,:]))
        plt.colorbar()
        plt.pause(1)


        # SAVE FILE HERE
        fullfileout = "{:}/ID_{:}".format(output_folder, i)
        np.savez(fullfileout, protein=protein, pssm=pssm, dca=dca, r1=r1[org_id].T, r2=r2[org_id].T, r3=r3[org_id].T)

        print("hes")




