import glob
import os

from numpy.linalg import svd, eigh
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
import copy

from src.utils import check_symmetric, Timer
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
    V_values = w[np.arange(nb)[:, None],idxs]
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
    return V[:,:,:k],V_values[...,:k]


if __name__ == "__main__":
    # cov_folder = "./../data/cov/"
    cov_folder = "F:/dataset_2d_small_validate/"
    # main_dataset_folder = "./../data/clean_pnet_test/"

    output_folder_1d_separated = "F:/small_dataset_1d_separete_diag_validate/"
    output_folder_1d_together = "F:/small_dataset_1d_validate/"
    report_iter = 500
    k_cov = 49
    k_contact = 49
    dt_save = np.float32

    os.makedirs(output_folder_1d_separated, exist_ok=True)
    os.makedirs(output_folder_1d_together, exist_ok=True)
    # Read MSA_folder in one at a time and match them up against main
    search_command = cov_folder + "*.npz"
    cov_files = [f for f in glob.glob(search_command)]
    t1 = time.time()
    for i, cov_file in enumerate(cov_files):
        if (i+1) % 1000 == 0:
            print("Done {:}, time {:2.2f}".format(i+1,time.time()-t1))
        data = np.load(cov_file, allow_pickle=True)
        seq = data['seq']
        r1 = data['r1']
        r2 = data['r2']
        r3 = data['r3']
        pssm = data['pssm']
        entropy = data['entropy']
        seq_len = len(seq)
        cov = data['cov2d']
        contact = data['contact2d']

        # cov2d = copy.deepcopy(cov)
        # contact2d = copy.deepcopy(contact)

        contact1d_vectors, contact1d_values = extract_1d_features(contact[None,:,:], k=k_contact, debug=False, safety=True, show_figures=1)
        cov1d_vectors,cov1d_values = extract_1d_features(cov, k=k_cov, debug=False, safety=True, show_figures=20)

        fullfileout = "{:}/ID_{:}".format(output_folder_1d_together, i)
        np.savez(fullfileout, seq=np.int32(seq), pssm=dt_save(pssm), r1=dt_save(r1), r2=dt_save(r2), r3=dt_save(r3), cov1d_vectors=dt_save(cov1d_vectors), cov1d_values=dt_save(cov1d_values), contact1d_vectors=dt_save(contact1d_vectors),  contact1d_values=dt_save(contact1d_values), entropy=dt_save(entropy))

        l = cov.shape[-1]
        tmp = np.arange(l)
        cov_diag = cov[:,tmp,tmp]
        contact_diag = contact[tmp,tmp]
        cov[:,tmp,tmp] = 0
        contact[tmp,tmp] = 0

        contact1d_vectors, contact1d_values = extract_1d_features(contact[None,:,:], k=k_contact, debug=False, safety=True, show_figures=1)
        cov1d_vectors,cov1d_values = extract_1d_features(cov, k=k_cov, debug=False, safety=True, show_figures=20)

        # contact1d = extract_1d_features(contact[None,:,:], k=k_contact, debug=True, safety=True, show_figures=5,print_all='./../figures/contact/')
        # cov1d = extract_1d_features(cov, k=k_cov, debug=True, safety=True, show_figures=0,print_all='./../figures/covariance/')

        contact1d = np.concatenate((contact_diag[None,:,None],contact1d_vectors),axis=2)
        cov1d = np.concatenate((cov_diag[:,:,None],cov1d_vectors),axis=2)

        fullfileout = "{:}/ID_{:}".format(output_folder_1d_separated, i)
        np.savez(fullfileout, seq=np.int32(seq), pssm=dt_save(pssm), r1=dt_save(r1), r2=dt_save(r2), r3=dt_save(r3), cov1d_vectors=dt_save(cov1d_vectors), cov1d_values=dt_save(cov1d_values), contact1d_vectors=dt_save(contact1d_vectors),  contact1d_values=dt_save(contact1d_values), entropy=dt_save(entropy))


        # np.savez(fullfileout, seq=np.int32(protein), pssm=dt_save(pssm), cov2d=dt_save(cov), contact2d=dt_save(contact2d), r1=dt_save(r1), r2=dt_save(r2), r3=dt_save(r3), cov1d=dt_save(cov1d), contact1d=dt_save(contact1d), entropy=dt_save(entropy))


        # 2D dataset, with only the smart points

        # SAVE FILE HERE