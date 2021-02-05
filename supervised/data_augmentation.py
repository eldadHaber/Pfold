import glob
import os
import time

import numpy as np
import torch
from scipy.ndimage import maximum_filter

from supervised.dataloader_utils import ConvertCoordToDists, convert_seq_to_onehot, convert_seq_to_onehot_torch
from supervised.network_transformer import tr2DistSmall
import matplotlib.pyplot as plt
import matplotlib

from supervised.visualization import cutfullprotein_simple

matplotlib.use('Agg')

def distPenality(D,dc=0.379,M=torch.ones(1)):
    U = torch.triu(D,2)
    p2 = torch.norm(M*torch.relu(2*dc - U))**2

    return p2

def compute_cost_matrix(iD2,min_subprotein_len):
    a = torch.cumsum(torch.cumsum(iD2[0,:,:], dim=1),dim=0)
    n = iD2.shape[-1]
    costM = torch.zeros((n,n),device=iD2.device)
    j = torch.arange(n)
    denom = torch.empty_like(a)
    for i in range(n):
        if i != 0:
            k = i-1
            costM[i,:] = a[:,-1] - torch.diagonal(a,0) - (a[k,-1] - a[k,:]) + \
                a[-1,:] - torch.diagonal(a,0) - (a[-1,k] - a[:,k]) + \
                a[:,k] - a[k,k] + \
                a[k,:] - a[k,k]
        else:
            costM[i,:] = a[:,-1] + a[-1,:] - 2 * torch.diagonal(a,0)
        denom[i, :] = (j - (i - 1)) * (-j + (n + i - 1))

    costM /= denom
    idx = torch.triu_indices(n,n, offset=min_subprotein_len)
    m2 = torch.zeros((n,n),dtype=torch.float32,device=costM.device)
    m2[idx[0,:],idx[1,:]] = 1
    tmp = torch.isnan(costM)
    costM[tmp] = 0
    costM = costM * m2
    return costM


def find_minima(A,max_cost,min_peak_dist):
    """
    This function will take a cost matrix and find all the minima in it that are below the max_cost value.
    This is done by subtracting the max_cost from the matrix and finding all minima with a value below 0.
    """
    n = A.shape[-1]
    idx = np.triu_indices(n, k=min_subprotein_len-1)
    m2 = np.zeros((n,n),dtype=np.bool)
    m2[idx] = True

    A = A-max_cost

    all_peaks = maximum_filter(-A, size=min_peak_dist) == -A
    peaks_mask = all_peaks * m2

    # We find the index of the peaks
    tmp = peaks_mask.nonzero()
    peaks_idx = np.asarray([tmp[0],tmp[1]])

    # And their cost
    peaks_cost = A[peaks_idx[0,:],peaks_idx[1,:]]

    # Next we sort the peaks according to their cost
    idx = np.argsort(peaks_cost)
    peak_idx = peaks_idx[:,idx]
    peak_cost = A[peak_idx[0,:],peak_idx[1,:]]

    # Finally we only select the peaks that are below the threshold
    m4 = peak_cost <= 0
    peak_idx = peak_idx[:,m4]
    return peak_idx


def find_local_maximums(A,search_mask):
    """
    A is a square matrix.
    A is a boolean mask, that tells which parts of A to search for peaks.
    """
    n = A.shape[0]
    p = np.zeros((n,n),dtype=np.bool)
    for i in range(n):
        for j in range(n):
            if search_mask[i,j]==False:
                continue
            ispeak = True
            if i != 0:
                ispeak *= A[i, j] > A[i - 1, j]
                if j!=0:
                    ispeak *= A[i, j] > A[i - 1, j - 1]
                if j!=n-1:
                    ispeak *= A[i, j] > A[i - 1, j + 1]
            if i != n-1:
                ispeak *= A[i, j] > A[i + 1, j]
                if j != 0:
                    ispeak *= A[i, j] > A[i + 1, j - 1]
                if j != n - 1:
                    ispeak *= A[i, j] > A[i + 1, j + 1]
            if j !=0:
                ispeak *= A[i, j] > A[i , j-1]
            if j !=n-1:
                ispeak *= A[i,j] > A[i,j+1]
            p[i,j] = ispeak
    return p

def predicting_missing_coordinates(seq,r,net):
    pad_mod = 8
    mask_pred = (r[0, :] == 0)
    if torch.sum(r[0,:] == 0) > 0: # We see whether there are any unknown coordinates in the protein that we need to estimate. Otherwise we just return r
        features = ()
        features += (convert_seq_to_onehot_torch(seq),)
        mask_known = ~ mask_pred
        features += (r, mask_known[None,:].float())
        features = torch.cat(features,dim=0).to(device,dtype=torch.float32)

        #Now we need to extend the sequence to modulo 8.
        nf, n = features.shape
        n2 = int(pad_mod * np.ceil(n / pad_mod))
        f_ext = torch.zeros((nf,n2),dtype=torch.float32,device=r.device)
        f_ext[:,:n] = features
        mask_pad = torch.ones(n2,dtype=torch.int64,device=r.device)
        mask_pad[n:] = 0


        with torch.no_grad():
            _, coords_pred_ext = net(f_ext[None,:,:],mask_pad[None,:])
        coords_pred = coords_pred_ext[0][:,:n]

        # x = distConstraint(coords_pred)

        r[:,mask_pred] = coords_pred[:,mask_pred].to(dtype=r.dtype)

        # dX = r[:, 1:] - r[:, :-1]
        # d = torch.sqrt(torch.sum(dX**2,dim=0,keepdim=True))
    else:
        pass
    return r, mask_pred


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    # inpainter = './../results/pretrained_networks/inpaint_400k.pt'
    # pnetfile = './../data/casp11/training_90'
    # output_folder = './../results/figures/data_aug/training2/'
    # pnetfile = './../data/casp11/validation'
    # output_folder = './../results/figures/data_aug/val1.5/'
    output_folder = './../data/casp11_training_90_fully_mapped_no_sub_augmented/'
    os.makedirs(output_folder, exist_ok=True)
    min_subprotein_len = 20
    # max_seq_len = 1000
    # min_seq_len = 20
    # min_ratio = 1
    min_peak_dist = 5
    cost_len_adjustment_slope = -0.6053
    # cost_len_adjustment_slope2 = -0.5872
    cost_adjustment_constant = np.exp(-1.881)
    cost_unit_scale = -10
    # cost_adjustment_constant2 = np.exp(-1.995)
    # _, net, _, _ =load_checkpoint(inpainter,device=device)
    # net.eval()
    save_3d_figures = False
    save_cost_matrix = False


    # args, log_units, AA_DICT, _,_,_,_ = parse_pnet(pnetfile, min_seq_len=min_seq_len, max_seq_len=max_seq_len, use_entropy=True, use_pssm=True,
    #                                       use_dssp=False, use_mask=False, use_coord=True, min_ratio=1)
    # minsep = Loss_reg_min_separation(-10)
    # ids = args['id']
    # rCas = args['rCa']
    # rCbs = args['rCb']
    # rNs = args['rN']
    #
    # pssms = args['pssm']
    # entropys = args['entropy']
    # seqs = args['seq']
    # seqs_len = args['seq_len']
    # nb = len(rCas)
    # fig = plt.figure(1, figsize=[20, 10])
    # AA_LIST = list(AA_DICT)
    #
    # cost_all = []

    t0 = time.time()
    folder = './../data/casp11_training_90_fully_mapped_no_sub/'
    search_command = folder + "*.npz"
    files = [f for f in glob.glob(search_command)]

    nfiles = len(files)
    nsubs = 0

    for ii in range(nfiles):
        t1 = time.time()
        dat = np.load(files[ii])
        seq_id = dat['id']
        seq_numpy = dat['seq']
        seq = torch.from_numpy(seq_numpy).to(device,dtype=torch.int64)
        r_numpy = dat['rCa']
        r = torch.from_numpy(r_numpy).to(device,dtype=torch.float64)
        log_units = dat['log_units']
        AA_LIST = dat['AA_LIST']

        # r, mask_pred = predicting_missing_coordinates(seq, r, net)
        t2 = time.time()
        n = r.shape[-1]
        scaling = 10.0 ** (log_units-cost_unit_scale)
        max_peak_cost = 0.5 * n**cost_len_adjustment_slope * cost_adjustment_constant * scaling**2

        D = tr2DistSmall(r[None,:,:])
        D2 = D**2
        iD2 = 1/D2
        idx = D == 0
        iD2[idx] = 0

        t3 = time.time()
        costM = compute_cost_matrix(iD2, 0)
        t4 = time.time()
        peaks_idx = find_minima(costM.cpu().numpy(), max_peak_cost, min_peak_dist)
        t5 =time.time()
        if save_cost_matrix:
            costM_adj = costM.cpu() - max_peak_cost
            m = costM.cpu() == 0
            costM_adj = costM.cpu() - max_peak_cost
            costM_adj[m] = 0
            costM_adj[0,-1] = torch.min(costM_adj)

            plt.clf()
            plt.subplot(1,2,1)
            plt.imshow((iD2[0,:,:]).cpu())
            plt.title("Inverse square distance")
            plt.colorbar()

            plt.subplot(1,2,2)
            plt.imshow(costM_adj)
            plt.title("Cost of subprotein")
            plt.colorbar()
            plt.savefig("{:}{:}".format(output_folder,ii))
        npeaks = peaks_idx.shape[-1]
        nsubs += npeaks - 1
        t6 = time.time()
        if save_3d_figures:
            for i in range(min(npeaks,500)):
                cutfullprotein_simple(r.cpu().numpy(),peaks_idx[0,i],peaks_idx[1,i],"{:}{:}_cut{:}".format(output_folder,ii,i))
        t7 = time.time()
        # print("{:}, length={:}, n_peaks={:} time taken={:2.2f}s, total time = {:2.2f}h, eta={:2.2f}h".format(ii,n,npeaks,t7-t1,(t7-t0)/3600,(t7-t0)/(ii+1)*(nfiles-(ii+1))/3600))
        # print("{:2.2f}s  {:2.2f}s  {:2.2f}s  {:2.2f}s  {:2.2f}s  {:2.2f}s".format(t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6))
        filename = "{:}{:}".format(output_folder,seq_id)
        np.savez(file=filename, seq=seq_numpy, rCa=r_numpy, id=seq_id, log_units=log_units, AA_LIST=AA_LIST, weight=1)
        weight = 1/npeaks
        for i in range(1,npeaks):
            idx0 = peaks_idx[0,i]
            idx1 = peaks_idx[1,i]
            seqi = seq_numpy[idx0:idx1]
            coords = r_numpy[:,idx0:idx1]
            filename = "{:}{:}_sub{:}".format(output_folder,seq_id,i)
            np.savez(file=filename, seq=seqi, rCa=coords, id=seq_id, log_units=log_units, AA_LIST=AA_LIST, weight=weight)
        print("{:}, subproteins={:} total time = {:2.2f}h, eta={:2.2f}h".format(ii,nsubs,(t7-t0)/3600,(t7-t0)/(ii+1)*(nfiles-(ii+1))/3600))

    # np.savez("{:}costmatrices".format(output_folder), seqs_len=seqs_len, cost_all=cost_all)

    print("End")
