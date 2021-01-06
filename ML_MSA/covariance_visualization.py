import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

covfile = './../figures/cov_10078.pt'
cov = torch.load(covfile)
cov_basename = (covfile.split("/")[-1]).split(".")[0]
covreffile = './../figures/cov_ref_10078.pt'
covref_basename = (covreffile.split("/")[-1]).split(".")[0]
cov_ref = torch.load(covreffile)
nA = cov.shape[-1]
l = cov.shape[0]

#NOTE THAT MOST OF THE AXES ARE WRONG HERE, they plot it as a matrix, meaning the first coord i rows in a matrix (y-axis)


cov = cov.cpu()
cov_ref = cov_ref.cpu()
for i in range(l):
    cov[i,i,:,:] = 1/nA
# So we end up with cov of shape (l,l,nA,nA)
# where the first dimension is the masked residue,
# the second dimension is the one it is compared to,
# which is set the value in the third dimension,
# giving the probability in the 4th dimension
#
# # FIRST WE VIZ THE REFERENCE
# fig = plt.figure(figsize=(15, 10))
# for i in range(nA):
#     plt.clf()
#     plt.imshow((cov_ref[:, :, i].cpu()))
#     plt.title("cov_ref amino acid {:}".format(i))
#     plt.colorbar()
#     save = "./../figures/cov_ref_{:}_{:}".format(covref_basename, i)
#     plt.savefig("{:}.png".format(save))
#
# #NEXT WE DO THE COVARIANCE.
# fig = plt.figure(figsize=(15, 10))
# for i in range(nA):
#     for j in range(nA):
#         plt.clf()
#         plt.imshow((cov[:, :, i, j].cpu()))
#         plt.title(
#             "Given amino acid i={:}, the probability for amino acid j={:} for all residue combinations.".format(i, j))
#         plt.xlabel("masked residue")
#         plt.ylabel("paired residue")
#         plt.colorbar()
#         save = "./../figures/cov_{:}_{:}_{:}".format(cov_basename, i, j)
#         plt.savefig("{:}.png".format(save))
#         # plt.pause(1)


fig = plt.figure(figsize=(15, 10))
for i in range(l):
    for j in range(nA):
        plt.clf()
        tmp=cov[i, :, j, :].cpu()
        tmp_sel = tmp[torch.arange(l)!=i,:]
        vmin = torch.min(tmp_sel)
        vmax = torch.max(tmp_sel)
        plt.imshow(tmp,vmin=vmin, vmax=vmax)
        plt.title(
            "Show the probabilities for masked residue i={:} compared to all other residues on amino acid j={:}".format(i, j))
        plt.xlabel("probabilities")
        plt.ylabel("paired residue")
        plt.colorbar()
        save = "./../figures/{:}_{:}_{:}".format(cov_basename, i, j)
        plt.savefig("{:}.png".format(save))
        # plt.pause(1)


fig = plt.figure(figsize=(15, 10))
for i in range(l):
    for j in range(nA):
        plt.clf()
        tmp=cov[i, :, j, :].cpu()-cov_ref[i,:,:]
        tmp_sel = tmp[torch.arange(l)!=i,:]
        vmin = torch.min(tmp_sel)
        vmax = torch.max(tmp_sel)
        plt.imshow((cov[i, :, j, :].cpu()-cov_ref[i,:,:]),vmin=vmin, vmax=vmax)
        plt.title(
            "Show the probabilities for masked residue i={:} compared to all other residues on amino acid j={:}".format(i, j))
        plt.xlabel("paired residue")
        plt.ylabel("probabilities")
        plt.colorbar()
        save = "./../figures/cov_adj_{:}_{:}_{:}".format(cov_basename, i, j)
        plt.savefig("{:}.png".format(save))
        # plt.pause(1)


#
#
# for i in range(l):
#     for j in range(l):
#         plt.clf()
#         plt.imshow((cov[i, j, :, :].cpu()))
#         plt.title("residue i={:},j={:}".format(i, j))
#         plt.title("Residue i={:} is masked, and residue j={:} is set to the following amino acid.".format(i, j))
#         plt.xlabel("amino acid of residue j")
#         plt.ylabel("predicted amino acid of i")
#         plt.colorbar()
#         save = "./../figures/residue_cov_{:}_{:}_{:}".format(cov_basename, i, j)
#         plt.savefig("{:}.png".format(save))
#         # plt.pause(1)


# NEXT WE PLOT THE ADJUSTED COVARIANCE
#
# fig = plt.figure(figsize=(15, 10))
# for j in range(nA):
#     for i in range(nA):
#         plt.clf()
#         plt.imshow((cov[:, :, i, j]-cov_ref[:,:,i]))
#         plt.title(
#             "Given amino acid i={:}, the probability for amino acid j={:} for all residue combinations.".format(i, j))
#         plt.xlabel("masked residue")
#         plt.ylabel("paired residue")
#         plt.colorbar()
#         save = "./../figures/cov_adj_{:}_{:}_{:}".format(cov_basename, i, j)
#         plt.savefig("{:}.png".format(save))
#         # plt.pause(1)
#
#
# for i in range(l):
#     for j in range(l):
#         plt.clf()
#         plt.imshow((cov[i, j, :, :].cpu()-cov_ref[i,j,:,None]))
#         plt.title("residue i={:},j={:}".format(i, j))
#         plt.title("Residue i={:} is masked, and residue j={:} is set to the following amino acid.".format(i, j))
#         plt.xlabel("amino acid of residue j")
#         plt.ylabel("predicted amino acid of i")
#         plt.colorbar()
#         save = "./../figures/residue_cov_adj_{:}_{:}_{:}".format(cov_basename, i, j)
#         plt.savefig("{:}.png".format(save))
#         # plt.pause(1)

