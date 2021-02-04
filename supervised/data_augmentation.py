import os
import time

import numpy as np
import pylab
import torch
import networkx as nx
from scipy.ndimage import maximum_filter

from supervised.IO import load_checkpoint
from supervised.dataloader_pnet import parse_pnet
from supervised.dataloader_utils import ConvertCoordToDists, convert_seq_to_onehot, convert_seq_to_onehot_torch
from supervised.loss import Loss_reg_min_separation
from supervised.network_transformer import tr2DistSmall
import matplotlib.pyplot as plt
import scipy.ndimage as ip
from skimage.feature import peak_local_max
import matplotlib

from supervised.vizualization_for_print import setup_print_figure

matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy.signal import find_peaks

a = torch.zeros((3,3))
for i in range(3):
    for j in range(3):
        a[i,j] = i*3+j+1


def distPenality(D,dc=0.379,M=torch.ones(1)):
    U = torch.triu(D,2)
    p2 = torch.norm(M*torch.relu(2*dc - U))**2

    return p2



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

        # x = distConstraint(coords_pred)

        r[:,mask_pred] = coords_pred[:,mask_pred].to(dtype=r.dtype)

        dX = r[:, 1:] - r[:, :-1]
        d = torch.sqrt(torch.sum(dX**2,dim=0,keepdim=True))



    else:
        pass
    return r, mask_pred


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
inpainter = './../results/pretrained_networks/inpaint_400k.pt'
# pnetfile = './../data/casp11/training_90'
# output_folder = './../results/figures/data_aug/training2/'
pnetfile = './../data/casp11/validation'
output_folder = './../results/figures/data_aug/validation3/'
os.makedirs(output_folder, exist_ok=True)
min_subprotein_len = 20
min_ratio = 1
cost_len_adjustment_slope = -0.6053
cost_adjustment_constant = np.exp(-1.881)

_, net, _, _ =load_checkpoint(inpainter,device=device)
net.eval()

args, log_units, AA_DICT, _,_,_,_ = parse_pnet(pnetfile, min_seq_len=-1, max_seq_len=1000, use_entropy=True, use_pssm=True,
                                      use_dssp=False, use_mask=False, use_coord=True, min_ratio=1)
minsep = Loss_reg_min_separation(-10)
ids = args['id']
rCas = args['rCa']
rCbs = args['rCb']
rNs = args['rN']

pssms = args['pssm']
entropys = args['entropy']
seqs = args['seq']
seqs_len = args['seq_len']
nb = len(rCas)
fig = plt.figure(1, figsize=[20, 10])
AA_LIST = list(AA_DICT)
t0 = time.time()

cost_all = []


for ii in range(nb):
    t1 = time.time()
    seq_id = ids[ii]
    rCa = rCas[ii].swapaxes(0, 1) * 10
    seq = torch.from_numpy(seqs[ii]).to(device,dtype=torch.float64)
    r = torch.from_numpy(rCa).to(device,dtype=torch.float64)
    r, mask_pred = predicting_missing_coordinates(seq, r, net)
    t2 = time.time()
    n = rCa.shape[-1]
    max_peak_cost = 0.6 * n**cost_len_adjustment_slope * cost_adjustment_constant

    D = tr2DistSmall(r[None,:,:])

    # For now we limit how small a distance in D can be.
    m1 = (D > 0).float()
    m2 = (D < minsep.d_mean).float()
    M = (m1 * m2).bool()
    D[M] = minsep.d_mean

    D2 = D**2
    iD2 = 1/D2
    idx = D == 0
    iD2[idx] = 0
    # plt.figure()
    # plt.imshow(D[0,:,:].cpu())
    # plt.colorbar()
    # plt.pause(1)
    #
    # plt.figure()
    # plt.imshow(iD2[0,:,:])
    # plt.colorbar()
    # plt.pause(1)

    n = iD2.shape[-1]
    t3 = time.time()
    costM = compute_cost_matrix_fast2(iD2, 0)
    t4 = time.time()

    # cost_all.append(costM.cpu().numpy())
    A = costM.cpu().numpy()
    m1 = A > 0
    idx = np.triu_indices(n, k=min_subprotein_len)
    m2 = np.zeros_like(m1)
    m2[idx] = 1
    m3 = m1 * m2
    # peaks = find_local_maximums(-A, m3)
    local_max = maximum_filter(-A, size=5) == -A
    peaks = local_max * m2


    tmp = peaks.nonzero()
    peak_idx = np.asarray([tmp[0],tmp[1]])


    peak_cost = A[peak_idx[0,:],peak_idx[1,:]]

    idx = np.argsort(peak_cost)
    peak_idx = peak_idx[:,idx]
    #
    #
    peak_cost = A[peak_idx[0,:],peak_idx[1,:]]
    m4 = peak_cost < max_peak_cost
    peak_idx = peak_idx[:,m4]
    #
    #
    # costM2 = costM.clone().cpu()
    # costM2[peak_idx[0,:],peak_idx[1,:]] = -1
    t5 =time.time()
    # # tmp = peak_local_max(-A, min_distance=5,exclude_border=False)
    # # tmp = ip.maximum_filter(-A, 2)
    # # msk = (-A == tmp)
    # # idx = m2[tmp[:,0],tmp[:,1]]
    # #
    # # peaks=tmp[idx]
    #
    #
    #
    # # constrain_and_find_regional_mins(costM.cpu().numpy(), mindist=2)
    setup_print_figure()
    font_size = 24  # Adjust as appropriate.
    fig = plt.figure(1,dpi=200)
    plt.clf()
    # pylab.axes([0.125,0.2,0.95-0.125,0.95-0.2])
    # pylab.plot(x,y1,'g:',label='$\sin(x)$')
    # pylab.plot(x,y2,'-b',label='$\cos(x)$')
    plt.imshow((iD2[0,:,:]).cpu())
    plt.title("Inverse square distance",fontdict={'fontsize': font_size})
    cb = plt.colorbar()

    # pylab.xlabel('$x$ (radians)')
    # pylab.ylabel('$y$')
    # pylab.legend()
    # plt.savefig("{:}ISD_{:}.eps".format(output_folder,ii))
    ax = plt.gca()
    cb.ax.tick_params(labelsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    # ax.tick_params(axis='both', which='minor', labelsize=16)
    plt.savefig("{:}ISD_{:}.png".format(output_folder,ii), bbox_inches='tight',dpi=600)

    setup_print_figure()
    fig = plt.figure(1,dpi=200)
    plt.clf()
    plt.imshow(costM.cpu())
    plt.title("Cut cost of subproteins",fontdict={'fontsize': font_size})
    cb = plt.colorbar()
    ax = plt.gca()
    cb.ax.tick_params(labelsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    # plt.savefig("{:}Cost_{:}.eps".format(output_folder,ii))
    plt.savefig("{:}Cost_{:}.png".format(output_folder,ii), bbox_inches='tight',dpi=600)

    setup_print_figure()
    fig = plt.figure(1,dpi=200)
    plt.clf()
    plt.title("3D view",fontdict={'fontsize': font_size})
    axes = plt.axes(projection='3d')
    # axes.set_xlabel("x")
    # axes.set_ylabel("y")
    # axes.set_zlabel("z")
    p = r.cpu().numpy()
    # axes.tick_params(axis='both', which='major', labelsize=font_size)
    color = np.zeros((n,3))
    color[:,0] = np.linspace(0,1,n)
    color[:,2] = np.linspace(1,0,n)
    axes.scatter(p[0, :], p[1, :], p[2, :], s=100, c=color, depthshade=True)
    axes.plot3D(p[0, :], p[1, :], p[2, :], 'gray', marker='')
    axes.grid(False)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_zticks([])
    axes.set_axis_off()
    # plt.pause(1)
    # pylab.title("Cost of subprotein")
    # pylab.colorbar()
    # plt.cool()
    # plt.savefig("{:}3d_view_{:}.eps".format(output_folder,ii))
    plt.savefig("{:}3d_view_{:}.png".format(output_folder,ii),bbox_inches='tight',dpi=600)

    #     M = (~ mask_pred).float()
#     MM = M[:,None] @ M[None,:]
#     plt.clf()
#     plt.subplot(1,3,1)
#     plt.imshow((iD2[0,:,:]*MM).cpu())
#     plt.title("Inverse square distance")
#     plt.colorbar()
#
#     plt.subplot(1,3,2)
#     plt.imshow(costM2.cpu())
#     plt.title("n peaks={:}".format(peak_idx.shape[-1]))
#     # plt.imshow((iD2[0,:,:]).cpu())
# #     plt.title("Inverse square distance with predictions")
#     plt.colorbar()
#
#     # fig = plt.figure(2, figsize=[20, 10])
#     # plt.imshow(MM[:,:])
#     # fig = plt.figure(4, figsize=[20, 10])
#     # plt.subplot(1,2,1)
#     # plt.imshow(costM.cpu())
#     # plt.title("Cost of subprotein")
#     # plt.colorbar()
#
#     plt.subplot(1,3,3)
#     plt.imshow(costM.cpu())
#     plt.title("Cost of subprotein")
#     plt.colorbar()
#     plt.savefig("{:}{:}".format(output_folder,ii))
    npeaks = peak_idx.shape[-1]
    # npeaks = 0
    t6 = time.time()
    # for i in range(min(npeaks,5)):
    #     cutfullprotein(r.cpu().numpy(),peak_idx[0,i],peak_idx[1,i],"{:}{:}_cut{:}".format(output_folder,ii,i),mask_pred.cpu().numpy())
    t7 = time.time()
    print("{:}, length={:}, n_peaks={:} time taken={:2.2f}s, total time = {:2.2f}h, eta={:2.2f}h".format(ii,n,npeaks,t7-t1,(t7-t0)/3600,(t7-t0)/(ii+1)*(nb-(ii+1))/3600))
    print("{:2.2f}s  {:2.2f}s  {:2.2f}s  {:2.2f}s  {:2.2f}s  {:2.2f}s".format(t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6))
    filename = "{:}{:}".format(output_folder,seq_id)
    seq_numpy = seq.cpu().numpy()
    np.savez(file=filename, seq=seq_numpy, peak_idx=peak_idx, rCa=rCas[ii].T, id=seq_id, log_units=log_units, AA_LIST=AA_LIST)
    for i in range(1,npeaks):
        idx0 = peak_idx[0,i]
        idx1 = peak_idx[1,i]
        seq_numpy = seq[idx0:idx1].cpu().numpy()
        coords = (rCas[ii].T)[:,idx0:idx1]
        filename = "{:}{:}_sub{:}".format(output_folder,seq_id,i)
        np.savez(file=filename, seq=seq_numpy, rCa=coords, id=seq_id, log_units=log_units, AA_LIST=AA_LIST)

np.savez("{:}costmatrices".format(output_folder), seqs_len=seqs_len, cost_all=cost_all)

print("End")
