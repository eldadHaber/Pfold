import os
import time

import numpy as np
import torch
import networkx as nx
from scipy.ndimage import maximum_filter

from supervised.IO import load_checkpoint
from supervised.dataloader_pnet import parse_pnet
from supervised.dataloader_utils import ConvertCoordToDists, convert_seq_to_onehot, convert_seq_to_onehot_torch
from supervised.network_transformer import tr2DistSmall
import matplotlib.pyplot as plt
import scipy.ndimage as ip
from skimage.feature import peak_local_max
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy.signal import find_peaks

a = torch.zeros((3,3))
for i in range(3):
    for j in range(3):
        a[i,j] = i*3+j+1

torch.nn.Dropout(p)

print("here")
a1 = torch.cumsum(a,dim=1)

a2 = torch.cumsum(a1,dim=0)

def compute_cost_matrix(iD2,Mblock,min_subprotein_len):
    n = iD2.shape[-1]
    costM = torch.zeros((n,n),device=device)
    for i in range(n):
        Mblock[:] = 0
        for j in range(i+min_subprotein_len,n): # We calculate proteins one shorter than acceptable, because we don't want to end with a lot of border cases in the peakfinder (here we mask them anyway)
            Mblock[i:j] = 1
            nMblock = (Mblock == 0).float()
            Mblock2 = mask_to_2d(Mblock)
            nMblock2 = mask_to_2d(nMblock)

            nnMblock2 = (nMblock2 == 0).int()
            BB = nnMblock2-Mblock2
            cost = torch.sum(iD2 * BB)
            costM[i,j] = cost
    return costM

#
# def compute_cost_matrix(iD2,Mblock,min_subprotein_len):
#     n = iD2.shape[-1]
#     costM = torch.zeros((n,n),device=device)
#     for i in range(n):
#         Mblock[:] = 0
#         for j in range(i+min_subprotein_len-1,n): # We calculate proteins one shorter than acceptable, because we don't want to end with a lot of border cases in the peakfinder (here we mask them anyway)
#             Mblock[i:j] = 1
#             nMblock = (Mblock == 0).float()
#             Mblock2 = mask_to_2d(Mblock)
#             nMblock2 = mask_to_2d(nMblock)
#
#             nnMblock2 = (nMblock2 == 0).int()
#             BB = nnMblock2-Mblock2
#             cost = torch.sum(iD2 * BB) / ((j-i)*(n-(j-i)))
#             costM[i,j] = cost
#     return costM
#
#


def compute_cost_matrix_fast(iD2,min_subprotein_len):
    a = torch.cumsum(torch.cumsum(iD2[0,:,:], dim=1),dim=0)
    n = iD2.shape[-1]
    costM = torch.zeros((n,n),device=device)
    for i in range(n):
        for j in range(i+min_subprotein_len,n):
            if i != 0:
                k = i-1
                costM[i,j] = a[j,-1] - a[j,j] - (a[k,-1] - a[k,j]) + \
                    a[-1,j] - a[j,j] - (a[-1,k] - a[j,k]) + \
                    a[j,k] - a[k,k] + \
                    a[k,j] - a[k,k]
            else:
                costM[i,j] = a[j,-1] - a[j,j] + a[-1,j] - a[j,j]
    return costM

def compute_cost_matrix_fast2(iD2,min_subprotein_len):
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

    # A = costM.cpu().numpy()
    # local_max = maximum_filter(-A, size=3) == -A
    #
    # fig = plt.figure(1, figsize=[20, 10])
    # plt.imshow(A)
    # plt.colorbar()
    # plt.savefig("test")
    #
    # plt.clf()
    # plt.imshow(local_max)
    # plt.colorbar()
    # plt.savefig("peaks")

        # for j in range(n):
        #     a[i,j] = (j-i+1)*(n-(j-i+1))

    costM /= denom
    idx = torch.triu_indices(n,n, offset=min_subprotein_len)
    m2 = torch.zeros((n,n),dtype=torch.float32,device=costM.device)
    m2[idx[0,:],idx[1,:]] = 1
    tmp = torch.isnan(costM)
    costM[tmp] = 0
    costM = costM * m2
    return costM





def cutfullprotein(rCa,cut1,cut2,filename,mask_pred):
    def rotate(angle):
        axes.view_init(azim=angle)

    fig = plt.figure(num=2, figsize=[15, 10])
    plt.clf()
    axes = plt.axes(projection='3d')
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")

    mask_state=mask_pred[0]
    protein_state = cut1 == 0
    segment_start = [0]
    segment_end = []
    for i in range(mask_pred.shape[0]):
        protein = i >= cut1 and i < cut2
        if mask_pred[i] != mask_state or protein_state != protein:
            mask_state = mask_pred[i]
            protein_state = protein
            segment_end.append(i+1)
            segment_start.append(i)
    segment_end.append(mask_pred.shape[0])

    for i, (seg_start,seg_end)  in enumerate(zip(segment_start,segment_end)):
        if mask_pred[seg_start]:
            if seg_start >= cut1 and seg_start < cut2:
                color = 'lightblue'
            else:
                color = 'lightpink'
        else:
            if seg_start >= cut1 and seg_start < cut2:
                color = 'blue'
            else:
                color = 'red'

        protein0, = axes.plot3D(rCa[0, seg_start:seg_end], rCa[1, seg_start:seg_end], rCa[2, seg_start:seg_end], color, marker='x')

    import matplotlib.patches as mpatches

    red_patch = mpatches.Patch(color='red', label='Remainder')
    blue_patch = mpatches.Patch(color='blue', label='Target')

    plt.legend(handles=[red_patch, blue_patch])

    angle = 3
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save('{:}.gif'.format(filename), writer=animation.PillowWriter(fps=20))

    return















#
#
#
#
# def cutfullprotein(rCa,cut1,cut2,filename,mask_pred):
#     def rotate(angle):
#         axes.view_init(azim=angle)
#
#     fig = plt.figure(num=2, figsize=[15, 10])
#     plt.clf()
#     axes = plt.axes(projection='3d')
#     axes.set_xlabel("x")
#     axes.set_ylabel("y")
#     axes.set_zlabel("z")
#
#     cut1 = 15
#     mask_pred [10] = True
#     cut2 = 35
#     mask_pred_ext = np.zeros_like(mask_pred)  #This is an ugly hack, to make sure the different parts are actually connected
#     # Just make sure to plot the unknown before the known if this is used.
#     for i in range(mask_pred.shape[0]):
#         if i == 0:
#             mask_pred_ext[i] = mask_pred[i] or mask_pred[i+1]
#         elif i == mask_pred.shape[0]-1:
#             mask_pred_ext[i] = mask_pred[i] or mask_pred[i-1]
#         else:
#             mask_pred_ext[i] = mask_pred[i-1] or mask_pred[i] or mask_pred[i+1]
#
#
#     n = rCa.shape[-1] # Number of amino acids in the protein
#     rest_known1 = ~mask_pred.copy()
#     rest_known1[cut1+1:] = False
#
#     rest_known2 = ~mask_pred.copy()
#     rest_known2[:cut2-1] = False
#
#     rest_unknown1 = mask_pred_ext.copy()
#     rest_unknown1[cut1+1:] = False
#
#     rest_unknown2 = mask_pred_ext.copy()
#     rest_unknown2[:cut2-1] = False
#
#     target_known = ~mask_pred.copy()
#     target_known[:cut1] = False
#     target_known[cut2:] = False
#
#     target_unknown = mask_pred_ext.copy()
#     target_unknown[:cut1] = False
#     target_unknown[cut2:] = False
#
#     # We start by plotting the target protein
#     protein0_ukn, = axes.plot3D(rCa[0, rest_unknown], rCa[1, rest_unknown], rCa[2, rest_unknown], 'salmon', marker='x')
#     protein0_kn, = axes.plot3D(rCa[0, rest_known], rCa[1, rest_known], rCa[2, rest_known], 'red', marker='x')
#     protein1_ukn, = axes.plot3D(rCa[0, target_unknown], rCa[1, target_unknown], rCa[2, target_unknown], 'lightblue', marker='x')
#     protein1_kn, = axes.plot3D(rCa[0, target_known], rCa[1, target_known], rCa[2, target_known], 'blue', marker='x')
#
#     plt.legend((protein0_kn, protein1_kn), ('Remainder', 'Target'))
#     angle = 3
#     ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
#     ani.save('{:}.gif'.format(filename), writer=animation.PillowWriter(fps=20))
#
#     # elev = 90
#     # azim = 0
#     # axes.view_init(elev=elev, azim=azim)
#     # save = "{}elev{}_azim{}.png".format(filename,elev,azim)
#     # fig.savefig(save)
#     #
#     # elev = 60
#     # azim = 20
#     # axes.view_init(elev=elev, azim=azim)
#     # save = "{}elev{}_azim{}.png".format(filename,elev,azim)
#     # fig.savefig(save)
#     #
#     # elev = 45
#     # azim = 40
#     # axes.view_init(elev=elev, azim=azim)
#     # save = "{}elev{}_azim{}.png".format(filename,elev,azim)
#     # fig.savefig(save)
#     #
#     # elev = 5
#     # azim = 85
#     # axes.view_init(elev=elev, azim=azim)
#     # save = "{}elev{}_azim{}.png".format(filename,elev,azim)
#     # fig.savefig(save)
#
#     return



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
        r[:,mask_pred] = coords_pred[:,mask_pred]
    else:
        pass
    return r, mask_pred


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
inpainter = './../results/pretrained_networks/inpaint_200k_100k.pt'
pnetfile = './../data/casp11/validation'
output_folder = './../results/figures/data_aug/v1/'
os.makedirs(output_folder, exist_ok=True)
min_subprotein_len = 20
max_peak_cost = 0.3

_, net, _, _ =load_checkpoint(inpainter,device=device)
net.eval()

args, log_units, AA_DICT = parse_pnet(pnetfile, min_seq_len=-1, max_seq_len=1000, use_entropy=True, use_pssm=True,
                                      use_dssp=False, use_mask=False, use_coord=True)


ids = args['id']
rCas = args['rCa']
rCbs = args['rCb']
rNs = args['rN']
pssms = args['pssm']
entropys = args['entropy']
seqs = args['seq']
nb = len(rCas)
fig = plt.figure(1, figsize=[20, 10])
t0 = time.time()
for ii in range(nb):
    t1 = time.time()
    rCa = rCas[ii].swapaxes(0, 1) * 10
    seq = torch.from_numpy(seqs[ii]).to(device)
    r = torch.from_numpy(rCa).to(device,dtype=torch.float32)
    r, mask_pred = predicting_missing_coordinates(seq, r, net)
    t2 = time.time()

    D = tr2DistSmall(r[None,:,:])
    D2 = D**2
    iD2 = 1/D2
    idx = D == 0
    iD2[idx] = 0
    n = iD2.shape[-1]
    t3 = time.time()
    costM = compute_cost_matrix_fast2(iD2, 0)
    t4 = time.time()

    A = costM.cpu().numpy()
    m1 = A > 0
    idx = np.triu_indices(n, k=min_subprotein_len)
    m2 = np.zeros_like(m1)
    m2[idx] = 1
    m3 = m1 * m2
    peaks = find_local_maximums(-A, m3)
    # local_max = maximum_filter(-A, size=3) == -A


    tmp = peaks.nonzero()
    peak_idx = np.asarray([tmp[0],tmp[1]])


    peak_cost = A[peak_idx[0,:],peak_idx[1,:]]

    idx = np.argsort(peak_cost)
    peak_idx = peak_idx[:,idx]


    peak_cost = A[peak_idx[0,:],peak_idx[1,:]]
    m4 = peak_cost < max_peak_cost
    peak_idx = peak_idx[:,m4]


    costM2 = costM.clone()
    costM2[peak_idx[0,:5],peak_idx[1,:5]] = -1
    costM2[peak_idx[0,5:],peak_idx[1,5:]] = -0.8
    t5 =time.time()
    # tmp = peak_local_max(-A, min_distance=5,exclude_border=False)
    # tmp = ip.maximum_filter(-A, 2)
    # msk = (-A == tmp)
    # idx = m2[tmp[:,0],tmp[:,1]]
    #
    # peaks=tmp[idx]



    # constrain_and_find_regional_mins(costM.cpu().numpy(), mindist=2)
    M = (~ mask_pred).float()
    MM = M[:,None] @ M[None,:]
    plt.clf()
    plt.subplot(1,3,1)
    plt.imshow((iD2[0,:,:]*MM).cpu())
    plt.title("Inverse square distance")
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.imshow((iD2[0,:,:]).cpu())
    plt.title("Inverse square distance with predictions")
    plt.colorbar()

    # fig = plt.figure(2, figsize=[20, 10])
    # plt.imshow(MM[:,:])
    # fig = plt.figure(4, figsize=[20, 10])
    # plt.subplot(1,2,1)
    # plt.imshow(costM.cpu())
    # plt.title("Cost of subprotein")
    # plt.colorbar()

    plt.subplot(1,3,3)
    plt.imshow(costM2.cpu())
    plt.title("Cost of subprotein")
    plt.colorbar()
    plt.savefig("{:}{:}".format(output_folder,ii))
    npeaks = peak_idx.shape[-1]
    t6 = time.time()
    for i in range(min(npeaks,5)):
        cutfullprotein(r.cpu().numpy(),peak_idx[0,i],peak_idx[1,i],"{:}{:}_cut{:}".format(output_folder,ii,i),mask_pred.cpu().numpy())
    t7 = time.time()
    print("{:}, length={:}, n_peaks={:} time taken={:2.2f}s, total time = {:2.2f}h, eta={:2.2f}h".format(ii,n,npeaks,t7-t1,(t7-t0)/3600,(t7-t0)/(ii+1)*(nb-(ii+1))/3600))
    print("{:2.2f}s  {:2.2f}s  {:2.2f}s  {:2.2f}s  {:2.2f}s  {:2.2f}s".format(t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6))



print("End")
