import glob
from numpy.linalg import svd, eigh
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
import copy

from preprocessing.MSA_to_cov import setup_protein_comparison
from srcOld.utils import check_symmetric

# matplotlib.use('TkAgg') #TkAgg

def extract_1d_features(A,k=10,debug=False,safety=True, show_figures=-1, print_all=None):
    """
    Does an eigen decomposition of an array of symmetric 2D matrices (n,m,m), with n being the batch size, and mxm the matrix shape
    Returns the k eigenvectors with largest absolute eigenvalue.

    safety = True, ensures that the matrix is symmetric.
    debug = True, prints the mean and max error when using the k eigenvectors in comparison to the full 2D matrix.
        Furthermore if debug = True, it also enables the possibility of using the following:

        show_figures, if set to an integer n>0, will print n, random comparisons of the 2D matrix and the corresponding matrix made with k eigenvectors.
        print_all, if not None, will attempt to save all possible 2D matrix comparisons to the folder given by print_all.

    """
    t0 = time.time()
    nb = A.shape[0]
    if safety:
        assert check_symmetric(A)

    w, v = eigh(A)
    idxs = np.argsort(np.abs(w),axis=-1)[:, ::-1]

    V = v[np.arange(nb)[:, None], :, idxs].swapaxes(1, 2)
    if debug:
        W = np.empty((nb,k,k),dtype=A.dtype)
        for i in range(nb):
            W[i,:,:] = np.diag(w[i,idxs[i,:k]])
        A_pc = V[:, :, :k] @ W @ np.swapaxes(V[:, :, :k],axis1=-1, axis2=-2)

        A_dif = A - A_pc
        A_dif_percent = np.linalg.norm(A_dif,axis=(1,2)) / np.linalg.norm(A,axis=(1,2))*100
        print("1D feature extraction error= {:2.2f}%, max={:2.2f}%, time taken: {:2.2f}s".format(np.mean(A_dif_percent),np.max(A_dif_percent),time.time()-t0))

        if show_figures > 0:
            bi = np.random.choice(nb,min(nb,show_figures),replace=False)
            for i in bi:
                plt.figure(figsize=(15, 10))
                plt.subplot(1, 3, 1)
                plt.imshow(A[i,:,:])
                plt.title("org ID={:}".format(i))
                plt.colorbar()
                plt.subplot(1, 3, 2)
                plt.imshow(A_pc[i,:,:])
                plt.title("eig={:}".format(k))
                plt.colorbar()
                plt.subplot(1, 3, 3)
                plt.title("error={:2.2f}%".format(A_dif_percent[i]))
                plt.imshow(np.abs(A_dif[i,:,:]))
                plt.colorbar()

            plt.pause(0.5)

        if print_all:
            plt.figure(figsize=(15, 10))
            for i in range(nb):
                plt.clf()
                plt.subplot(1, 3, 1)
                plt.imshow(A[i,:,:])
                plt.title("org ID={:}".format(i))
                plt.colorbar()
                plt.subplot(1, 3, 2)
                plt.imshow(A_pc[i,:,:])
                plt.title("eig={:}".format(k))
                plt.colorbar()
                plt.subplot(1, 3, 3)
                plt.title("error={:2.2f}%".format(A_dif_percent[i]))
                plt.imshow(np.abs(A_dif[i,:,:]))
                plt.colorbar()
                plt.savefig("{:}/{:}.png".format(print_all,i))
    return V[:,:,:k]

if __name__ == "__main__":
    cov_folder = "./../data/cov/"
    main_dataset_folder = "./../data/clean_pnet_train/"
    output_folder = "./../data/train/"
    report_iter = 500
    k_cov = 9
    k_contact = 19

    # Read all of main_folder in
    search_command = main_dataset_folder + "*.npz"
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

    seqs_list, seqs_list_org_id, lookup = setup_protein_comparison(seqs, seqs_len)

    # Read MSA_folder in one at a time and match them up against main
    search_command = cov_folder + "*.npz"
    cov_files = [f for f in glob.glob(search_command)]
    for i, cov_file in enumerate(cov_files):
        data = np.load(cov_file, allow_pickle=True)
        dca = data['dca']
        protein = data['protein']
        seq_len = len(protein)
        try:
            seq_idx = lookup[seq_len]
        except:
            print("Protein length not in database")
            continue
        seqs_i = seqs_list[seq_idx]
        res = np.mean(protein == seqs_i, axis=1) == 1
        org_id = (seqs_list_org_id[seq_idx][res]).squeeze()
        if len(org_id) != 1:
            print("Protein matching problems. org_id= {:}".format(org_id))

        pssm = pssms[org_id]
        entropy = entropys[org_id]
        r1 = r1s[org_id]
        r2 = r2s[org_id]
        r3 = r3s[org_id]

        # Ensure dca is symmetric, which it was supposed to be
        dca = 0.5 * (dca + np.swapaxes(dca,0,1))

        cov = dca[:,:,:-1]
        contact = dca[:,:,-1]
        cov = np.swapaxes(cov,0,2)

        cov2d = copy.deepcopy(cov)
        contact2d = copy.deepcopy(contact)

        l = cov.shape[-1]
        tmp = np.arange(l)
        cov_diag = cov[:,tmp,tmp]
        contact_diag = contact[tmp,tmp]
        cov[:,tmp,tmp] = 0
        contact[tmp,tmp] = 0

        contact1d = extract_1d_features(contact[None,:,:], k=k_contact, debug=False, safety=True, show_figures=1)
        cov1d = extract_1d_features(cov, k=k_cov, debug=False, safety=True, show_figures=20)
        # contact_features = extract_1d_features(contact[None,:,:], k=k_contact, debug=True, safety=True, show_figures=5,print_all='./../figures/contact/')
        # cov_features = extract_1d_features(cov, k=k_cov, debug=True, safety=True, show_figures=0,print_all='./../figures/covariance')

        contact1d = np.concatenate((contact_diag[None,:,None],contact1d),axis=2)
        cov1d = np.concatenate((cov_diag[:,:,None],cov1d),axis=2)

        # SAVE FILE HERE
        fullfileout = "{:}/ID_{:}".format(output_folder, i)

        np.savez(fullfileout, protein=protein, pssm=pssm, cov2d=cov, contact2d=contact2d, r1=r1, r2=r2, r3=r3, cov1d=cov1d, contact1d=contact1d)


