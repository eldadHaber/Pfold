import copy
import matplotlib.pyplot as plt
import matplotlib
import time
from supervised.utils import check_symmetric, Timer
from numpy.linalg import svd, eigh

import numpy as np



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

                Am = np.max(A)
                A_pcm= np.max(A_pc)
                vmax = max(Am,A_pcm)

                Am = np.min(A)
                A_pcm= np.min(A_pc)
                vmin = min(Am,A_pcm)

                plt.clf()
                plt.imshow(A[i,:,:],vmin=vmin, vmax=vmax)
                plt.colorbar()
                prefix = 'org'
                plt.title("Original", fontsize=50)
                plt.savefig("{:}/{:}_{:}.png".format(print_all, i,prefix))

                plt.clf()
                plt.imshow(A_pc[i, :, :], vmin=vmin, vmax=vmax)
                prefix = k
                plt.title("k={:}".format(k), fontsize=50)
                plt.savefig("{:}/{:}_{:}.png".format(print_all, i, prefix))



    return V[:,:,:k]



if __name__ == '__main__':

    cov_file = 'D:/Dropbox/ComputationalGenetics/data/cov/off/ID_125.npz'
    k_contact = 50


    data = np.load(cov_file, allow_pickle=True)
    protein = data['protein']
    seq_len = len(protein)
    dca = data['dca']
    # Ensure dca is symmetric, which it was supposed to be
    dca = 0.5 * (dca + np.swapaxes(dca, 0, 1))

    cov = dca[:, :, :-1]
    contact = dca[:, :, -1]
    cov = np.swapaxes(cov, 0, 2)

    cov2d = copy.deepcopy(cov)
    contact2d = copy.deepcopy(contact)

    l = cov.shape[-1]
    tmp = np.arange(l)
    cov_diag = cov[:, tmp, tmp]
    contact_diag = contact[tmp, tmp]
    cov[:, tmp, tmp] = 0
    contact[tmp, tmp] = 0

    contact1d = extract_1d_features(contact[None,:,:], k=k_contact, debug=True, safety=True, show_figures=5,print_all='./../figures/contact/')
    print("here")

