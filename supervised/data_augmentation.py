import os
import time

import numpy as np
import torch
import networkx as nx
from scipy.ndimage import maximum_filter

from supervised.IO import load_checkpoint
from supervised.dataloader_pnet import parse_pnet
from supervised.dataloader_utils import ConvertCoordToDists, convert_seq_to_onehot
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
    for i in range(n):
        if i != 0:
            k = i-1
            costM[i,:] = a[:,-1] - torch.diagonal(a,0) - (a[k,-1] - a[k,:]) + \
                a[-1,:] - torch.diagonal(a,0) - (a[-1,k] - a[:,k]) + \
                a[:,k] - a[k,k] + \
                a[k,:] - a[k,k]
        else:
            costM[i,:] = a[:,-1] + a[-1,:] - 2 * torch.diagonal(a,0)

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


    idx = torch.triu_indices(n,n, offset=min_subprotein_len)
    m2 = torch.zeros((n,n),dtype=torch.float32,device=costM.device)
    m2[idx[0,:],idx[1,:]] = 1
    costM = costM * m2
    return costM





def cutfullprotein(rCa,cut1,cut2,filename):
    def rotate(angle):
        axes.view_init(azim=angle)

    fig = plt.figure(num=2, figsize=[15, 10])
    plt.clf()
    axes = plt.axes(projection='3d')
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")

    n = rCa.shape[-1] # Number of amino acids in the protein

    # We start by plotting the target protein
    protein1, = axes.plot3D(rCa[0, :cut1+1], rCa[1, :cut1+1], rCa[2, :cut1+1], 'red', marker='x')
    protein3, = axes.plot3D(rCa[0, cut2-1:], rCa[1, cut2-1:], rCa[2, cut2-1:], 'red', marker='x')
    protein2, = axes.plot3D(rCa[0, cut1:cut2], rCa[1, cut1:cut2], rCa[2, cut1:cut2], 'blue', marker='x')
    plt.legend((protein1, protein2), ('Remainder', 'Target'))
    angle = 3
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save('{:}.gif'.format(filename), writer=animation.PillowWriter(fps=20))

    # elev = 90
    # azim = 0
    # axes.view_init(elev=elev, azim=azim)
    # save = "{}elev{}_azim{}.png".format(filename,elev,azim)
    # fig.savefig(save)
    #
    # elev = 60
    # azim = 20
    # axes.view_init(elev=elev, azim=azim)
    # save = "{}elev{}_azim{}.png".format(filename,elev,azim)
    # fig.savefig(save)
    #
    # elev = 45
    # azim = 40
    # axes.view_init(elev=elev, azim=azim)
    # save = "{}elev{}_azim{}.png".format(filename,elev,azim)
    # fig.savefig(save)
    #
    # elev = 5
    # azim = 85
    # axes.view_init(elev=elev, azim=azim)
    # save = "{}elev{}_azim{}.png".format(filename,elev,azim)
    # fig.savefig(save)

    return


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


def mask_to_2d(M):
    return M[:, None] @ M[None, :]
#
# A = np.zeros((10,10))
# A[idx] = 1
# A[1,4] = 1
# A[3,6] = 1.5
# A[3,5] = 2.5
# A[4,9] = 3.5
# A[8,9] = 1
# tmp = ip.maximum_filter(A,2)
# tmp2 = peak_local_max(A,min_distance=2)
# fig = plt.figure(1, figsize=[20, 10])
# plt.imshow(A)
# plt.colorbar()
# plt.pause(1)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
inpainter = './../results/pretrained_networks/inpaint_test.pt'
pnetfile = './../data/casp11/validation'
output_folder = './../results/figures/data_aug/test/'
os.makedirs(output_folder, exist_ok=True)
min_subprotein_len = 20
max_peak_cost = 0.3

_, net, _, _ =load_checkpoint(inpainter,device=device)
net.eval()

args, log_units, AA_DICT = parse_pnet(pnetfile, min_seq_len=-1, max_seq_len=1000, use_entropy=True, use_pssm=True,
                                      use_dssp=False, use_mask=False, use_coord=True)


ids = args['id']
rCa = args['rCa']
rCb = args['rCb']
rN = args['rN']
pssm = args['pssm']
entropy = args['entropy']
seq = args['seq']
nb = len(rCa)
fig = plt.figure(1, figsize=[20, 10])
t0 = time.time()
for ii in range(nb):
    t1 = time.time()
    r_numpy = rCa[ii].swapaxes(0, 1)
    r = torch.from_numpy(r_numpy).to(device,dtype=torch.float32)
    mask_known_numpy = (r_numpy[0,:] != 0)
    mask_unknown_numpy = (r_numpy[0,:] == 0)
    mask_known = torch.from_numpy(mask_known_numpy).to(device,dtype=torch.float32)
    mask_unknown = torch.from_numpy(mask_unknown_numpy).to(device,dtype=torch.float32)

    if np.sum(r_numpy[0,:] == 0) > 0: # We see whether there are any unknown coordinates in the protein that we need to estimate.
        features_numpy = ()
        features_numpy += (convert_seq_to_onehot(seq[ii]),)
        features_numpy += (r_numpy, mask_known_numpy.astype(float)[None,:])
        features = torch.from_numpy(np.concatenate(features_numpy)).to(device,dtype=torch.float32)

        with torch.no_grad():
            mask_padding = torch.ones_like(mask_known)
            _, coords_pred = net(features[None,:,:],mask_padding[None,:])
        r[:,mask_unknown.bool()] = coords_pred[0][:,mask_unknown.bool()]

    M = mask_known
    MM = M[:,None] @ M[None,:]

    D = MM * tr2DistSmall(r[None,:,:])
    D2 = D**2
    iD2 = 1/D2
    idx = D2 == 0
    iD2[idx] = 0
    n = iD2.shape[-1]
    tt2 = time.time()
    costM = compute_cost_matrix_fast2(iD2, 0)
    tt3 = time.time()
    print("cost_matrix_fast={:2.2f}s".format(tt3-tt2))

    A = costM.cpu().numpy()
    m1 = A > 0
    idx = np.triu_indices(n, k=min_subprotein_len)
    m2 = np.zeros_like(m1)
    m2[idx] = 1
    m3 = m1 * m2
    peaks = find_local_maximums(-A, m3)
    peaks2, _ = find_peaks(-A, prominence=1)

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

    # tmp = peak_local_max(-A, min_distance=5,exclude_border=False)
    # tmp = ip.maximum_filter(-A, 2)
    # msk = (-A == tmp)
    # idx = m2[tmp[:,0],tmp[:,1]]
    #
    # peaks=tmp[idx]



    # constrain_and_find_regional_mins(costM.cpu().numpy(), mindist=2)

    # plt.clf()
    # plt.subplot(1,2,1)
    # plt.imshow((iD2[0,:,:]*MM).cpu())
    # plt.title("Inverse square distance")
    # plt.colorbar()
    # # fig = plt.figure(2, figsize=[20, 10])
    # # plt.imshow(MM[:,:])
    # # fig = plt.figure(4, figsize=[20, 10])
    # # plt.subplot(1,2,1)
    # # plt.imshow(costM.cpu())
    # # plt.title("Cost of subprotein")
    # # plt.colorbar()
    #
    # plt.subplot(1,2,2)
    # plt.imshow(costM2.cpu())
    # plt.title("Cost of subprotein")
    # plt.colorbar()
    # plt.savefig("{:}{:}".format(output_folder,ii))
    t2 = time.time()
    print("{:}, length={:}, time taken={:2.2f}s, total time = {:2.2f}h, eta={:2.2f}h".format(ii,n,t2-t1,(t2-t0)/3600,(t2-t0)/(ii+1)*(nb-(ii+1))/3600))
    # for i in range(min(peak_idx.shape[-1],5)):
    #     cutfullprotein(rCa[ii].T,peak_idx[0,i],peak_idx[1,i],"{:}{:}_cut{:}".format(output_folder,ii,i))



print("End")
