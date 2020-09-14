import time

import matplotlib
import numpy as np

matplotlib.use('Agg')

import torch
import torch.nn as nn

class LossMultiTargets(nn.Module):
    def __init__(self,loss_fnc=torch.nn.CrossEntropyLoss()):
        super(LossMultiTargets, self).__init__()
        self.loss = loss_fnc

    def forward(self, inputs,targets):
        # loss = []
        # for (input,target) in zip(inputs,targets):
        #     loss.append(self.loss(input,target))
        loss = 0
        nb = len(targets)
        for (input,target) in zip(inputs,targets):
            loss += self.loss(input,target)
        loss /= nb
        return loss

class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss,self).__init__()

    def forward(self, input, target):
        #We only want places where the target is larger than zero (remember this is for distances)
        # mask = target > 0
        # result = torch.mean((input[mask] - target[mask])**2)
        # result = torch.norm((input[mask] - target[mask])) ** 2 / torch.norm(target[mask]) ** 2
        nb = target.shape[0]
        result = 0
        for i in range(nb):
            inputi = input[i,:,:]
            targeti = target[i,:,:]
            maski = targeti > 0
            if torch.sum(maski) == 0: #nothing to learn from this one
                continue
            assert torch.norm(targeti[maski]) > 0
            result += torch.norm((inputi[maski] - targeti[maski])) ** 2 / torch.norm(targeti[maski]) ** 2

        return result/nb


def pc_translation_rotation_matching(r1,r2):
    '''
    Given two sets of 3D points of equal size. It computes the distance between these two sets of points, when allowing translation and rotation of the point clouds.
    We compute both chirality, and take whichever one has the lowest loss.
    r1 -> Tensor of shape (3,n)
    r2 -> Tensor of shape (3,n)
    '''

    #First we translate the two sets, by setting both their centroids to origin
    r1c = r1 - torch.mean(r1, dim=1, keepdim=True)
    r2c = r2 - torch.mean(r2, dim=1, keepdim=True)

    H = r1c @ r2c.transpose(0,1)
    t1 = time.time()

    U, S, V = torch.svd(H)

    t2 = time.time()

    d = torch.sign(torch.det(V @ U.transpose(0,1)))
    t3 = time.time()
    tmp = torch.diag_embed(torch.tensor([1, 1, d])).to(device=V.device)
    t4 = time.time()
    R = V @ tmp @ U.transpose(0,1)
    t5 = time.time()

    # tmp2 = torch.diag_embed(torch.tensor([1, 1, -d])).to(device=V.device)
    # R2 = V @ tmp2 @ U.transpose(0,1)

    r1cr = R @ r1c
    # r1cr2 = R2 @ r1c

    assert torch.norm(r2c) > 0
    loss_tr1 = torch.norm(r1cr - r2c) ** 2 / torch.norm(r2c) ** 2
    # loss_tr2 = torch.norm(r1cr2 - r2c) ** 2 / torch.norm(r2c) ** 2

    # if loss_tr1 < loss_tr2:
    loss_tr = loss_tr1
    # pred = r1cr.squeeze().cpu().detach().numpy()
    # else:
    #     pred = r1cr2.squeeze().cpu().detach().numpy()
    #     loss_tr = loss_tr2
    # target = r2c.squeeze().cpu().detach().numpy()
    print("{:2.4f},{:2.4f},{:2.4f},{:2.4f}".format(t2-t1,t3-t2,t4-t3,t5-t4))
    return loss_tr#, pred, target


def loss_tr_wrapper(r1,r2):
    '''

    Note that any point with r2 coordinates set to zero is considered masked and will not be included in the calculation. (so use r1 for prediction and r2 for target, and just make sure no target point are accidently zero. Remember the point cloud is translation invariant, so you can just translate all points if needed)
    '''

    nb = r1.shape[0]
    loss_tr = 0
    for i in range(nb):
        r1i = r1[i, :, :]
        r2i = r2[i,:,:]
        mask = (r2i != 0).reshape(3, -1)
        mask = torch.sum(mask,dim=0) > 0
        r1i = r1i[:,mask]
        r2i = r2i[:,mask]
        # loss_tri, predi, targeti = pc_translation_rotation_matching(r1i, r2i)
        loss_tri = pc_translation_rotation_matching(r1i, r2i)
        loss_tr += loss_tri
    loss_tr /= nb
    return loss_tr#, predi, targeti

def loss_tr(r1,r2, return_coords=False):
    t1 = time.time()
    loss_tr = 0
    mask = (r2 != 0).reshape(r2.shape)
    mask = (torch.sum(mask,dim=1) > 0).unsqueeze(1)
    mask = mask.repeat(1,3,1)
    batch_mask = torch.sum(mask,dim=(1,2)) > 0

    r1 = r1[batch_mask,:,:]
    r2 = r2[batch_mask,:,:]
    mask = mask[batch_mask,:,:]

    nb = r1.shape[0]


    t2 = time.time()
    #First we translate the two sets, by setting both their centroids to origin
    r1c = torch.empty_like(r1)
    r2c = torch.empty_like(r2)
    for i in range(nb):
        r1c[i, :, :] = r1[i, :, :] - torch.mean(r1[i, mask[i, :, :]].reshape(3, -1), dim=1, keepdim=True)
        r2c[i, :, :] = r2[i, :, :] - torch.mean(r2[i, mask[i, :, :]].reshape(3, -1), dim=1, keepdim=True)
    t3 = time.time()
    r1c = r1c * mask
    r2c = r2c * mask

    H = torch.bmm(r1c,r2c.transpose(1,2))
    # try:
    #     U, S, V = torch.svd(H)
    # except:  # torch.svd may have convergence issues for GPU and CPU.
    #     U, S, V = torch.svd(H + 1e-4 * H.mean() * torch.rand(H.shape,device=H.device))
    U, S, V = torch.svd(H)
    t4 = time.time()

    d = torch.sign(torch.det(torch.bmm(V, U.transpose(1,2))))
    t5 = time.time()

    tt=torch.tensor([[1]*nb, [1]*nb, d]).transpose(0,1)
    tmp = torch.diag_embed(tt).to(device=V.device)
    t6 = time.time()

    R = torch.bmm(V, torch.bmm(tmp, U.transpose(1,2)))

    r1cr = torch.bmm(R, r1c)

    loss_tr = torch.mean(torch.norm(r1cr - r2c, dim=(1, 2)) ** 2 / torch.norm(r2c, dim=(1, 2)) ** 2)
    t7 = time.time()
    # print("{:2.4f},{:2.4f},{:2.4f},{:2.4f},{:2.4f},{:2.4f}".format(t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6))
    if return_coords:
        pred = r1cr[-1,:,:].squeeze().cpu().detach().numpy()
        target = r2c[-1,:,:].squeeze().cpu().detach().numpy()
        return loss_tr, pred, target
    else:
        return loss_tr