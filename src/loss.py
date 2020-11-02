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
        mask = target > 0

        batch_mask = torch.sum(mask, dim=(1, 2)) > 0
        input = input[batch_mask, :, :]
        target = target[batch_mask, :, :]
        mask = mask[batch_mask, :, :]
        nb = target.shape[0]

        result = torch.sum(torch.norm(input * mask - target * mask,dim=(1,2)) ** 2 / torch.norm(target * mask,dim=(1,2)) ** 2)

        return result/nb


class EMSELoss(torch.nn.Module):
    """
    Exponential MSE loss
    """
    def __init__(self):
        super(EMSELoss,self).__init__()

    def forward(self, input, target, sigma=0.5):
        #We only want places where the target is larger than zero (remember this is for distances)
        mask = target > 0

        batch_mask = torch.sum(mask, dim=(1, 2)) > 0
        input = input[batch_mask, :, :]
        target = target[batch_mask, :, :]
        mask = mask[batch_mask, :, :]
        nb = target.shape[0]

        input_e = torch.exp(-input/sigma)
        target_e = torch.exp(-target/sigma)

        result = torch.sum(torch.norm(input_e * mask - target_e * mask,dim=(1,2)) ** 2 / torch.norm(target_e * mask,dim=(1,2)) ** 2)

        return result/nb



def loss_tr(r1,r2, return_coords=False):
    '''
    Given two sets of 3D points of equal size. It computes the distance between these two sets of points, when allowing translation and rotation of the point clouds.
    r1 -> Tensor of shape (b,3,n)
    r2 -> Tensor of shape (b,3,n)
    '''
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
    r1c = r1 - torch.sum(r1 * mask, dim=2, keepdim=True) / torch.sum(mask, dim=2, keepdim=True)
    r2c = r2 - torch.sum(r2 * mask, dim=2, keepdim=True) / torch.sum(mask, dim=2, keepdim=True)
    r1c = r1c * mask
    r2c = r2c * mask

    # r1c2 = r1 - torch.sum(r1 * mask, dim=2, keepdim=True) / torch.sum(mask, dim=2, keepdim=True)
    # r1c2 = r1c2 * mask

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


def loss_tr_tuples(r1s,r2s, return_coords=False):
    '''
    Given two sets of 3D points of equal size. It computes the distance between these two sets of points, when allowing translation and rotation of the point clouds.
    r1 -> Tensors of shape (b,3d,n)
    r2 -> Tuples of length d, containing Tensors of shape (b,3,n)
    '''
    loss_tr = 0
    coords_pred = ()
    coords_target = ()
    for i,r2 in enumerate(r2s):
        r1 = r1s[:,3*i:3*i+3,:]

        mask = (r2 != 0).reshape(r2.shape)
        mask = (torch.sum(mask,dim=1) > 0).unsqueeze(1)
        batch_mask = torch.sum(mask,dim=(1,2)) > 10 # There needs to be at least 2 points in a protein for it to make sense, 1 point is nothing. 2 Might also be causing trouble so now we try 20
        mask = mask.repeat(1,3,1)

        r1 = r1[batch_mask,:,:]
        r2 = r2[batch_mask,:,:]
        mask = mask[batch_mask,:,:]

        #First we translate the two sets, by setting both their centroids to origin
        r1c = r1 - torch.sum(r1 * mask, dim=2, keepdim=True) / torch.sum(mask, dim=2, keepdim=True)
        r2c = r2 - torch.sum(r2 * mask, dim=2, keepdim=True) / torch.sum(mask, dim=2, keepdim=True)
        r1c = r1c * mask
        r2c = r2c * mask

        H = torch.bmm(r1c,r2c.transpose(1,2))
        U, S, V = torch.svd(H)

        d = torch.sign(torch.det(torch.bmm(V, U.transpose(1,2))))

        ones = torch.ones_like(d)
        a = torch.stack((ones, ones, d), dim=-1)
        tmp = torch.diag_embed(a)

        R = torch.bmm(V, torch.bmm(tmp, U.transpose(1,2)))

        r1cr = torch.bmm(R, r1c)

        loss_tr += torch.mean(torch.norm(r1cr - r2c, dim=(1, 2)) ** 2 / torch.norm(r2c, dim=(1, 2)) ** 2)
        if return_coords:
            coords_pred += (r1cr[:,:].squeeze().cpu().detach().numpy(),)
            coords_target += (r2c[:,:].squeeze().cpu().detach().numpy(),)
    loss_tr / len(r2s)
    if return_coords:
        return loss_tr, coords_pred, coords_target
    else:
        return loss_tr




def loss_tr_all(r1,r2, return_coords=False):
    '''
    Given two sets of 3D points of equal size. It computes the distance between these two sets of points, when allowing translation and rotation of the point clouds.
    r2 -> Tuple of length d, where d=# different coordinates used (rN,rCa,rCb), each element in the tuple r2[0] has shape (b,3,n), b=batch, n=sequence length
    r1 -> Tensor of shape (b,3*d,n) d=# different coordinates (rN,rCa,rCb), b=batch, n=sequence length
    '''
    #First we pack the information into a more sensible structure
    r2=torch.stack(r2,dim=0)
    r1=r1.reshape(r2.shape)

    mask = (r2 != 0).reshape(r2.shape)
    mask = (torch.sum(mask,dim=2) > 0).unsqueeze(2)
    mask = mask.repeat(1,1,3,1)

    batch_mask = torch.sum(mask,dim=(2,3)) > 0

    # r1 = r1[batch_mask,:,:]
    # r2 = r2[batch_mask,:,:]
    # mask = mask[batch_mask,:,:]

    # nb = r1.shape[0]


    t2 = time.time()
    #First we translate the two sets, by setting both their centroids to origin
    r1c = r1 - torch.sum(r1 * mask, dim=3, keepdim=True) / torch.sum(mask, dim=3, keepdim=True)
    r2c = r2 - torch.sum(r2 * mask, dim=3, keepdim=True) / torch.sum(mask, dim=3, keepdim=True)
    r1c = r1c * mask
    r2c = r2c * mask
    #
    #
    # r1i = r1[2,:,:,:]
    # r2i = r2[2,:,:,:]
    # maski = mask[2,:,:,:]
    # r1ic = r1i - torch.sum(r1i * maski, dim=2, keepdim=True) / torch.sum(maski, dim=2, keepdim=True)
    # r2ic = r2i - torch.sum(r2i * maski, dim=2, keepdim=True) / torch.sum(maski, dim=2, keepdim=True)
    # r1ic = r1ic * maski
    # r2ic = r2ic * maski
    # Hi = torch.bmm(r1ic,r2ic.transpose(1,2))
    # Ui, Si, Vi = torch.svd(Hi)
    # di = torch.sign(torch.det(torch.bmm(Vi, Ui.transpose(1,2))))
    # nb = 5
    # tt=torch.tensor([[1]*nb, [1]*nb, di]).transpose(0,1)
    # tmpi = torch.diag_embed(tt).to(device=Vi.device)
    # Ri = torch.bmm(Vi, torch.bmm(tmpi, Ui.transpose(1,2)))
    # r1icr = torch.bmm(Ri, r1ic)
    # loss_tri = torch.norm(r1icr * maski - r2ic * maski, dim=(1, 2)) ** 2 / torch.norm(r2ic * maski, dim=(1, 2)) ** 2



    # r1c2 = r1 - torch.sum(r1 * mask, dim=2, keepdim=True) / torch.sum(mask, dim=2, keepdim=True)
    # r1c2 = r1c2 * mask
    H = r1c @ r2c.transpose(2,3)
    H = H * batch_mask[:,:,None,None]

    U, S, V = torch.svd(H)
    t4 = time.time()

    d = torch.sign(torch.det(V @ U.transpose(2,3)))
    t5 = time.time()

    ones = torch.ones_like(d)
    a=torch.stack((ones,ones,d),dim=-1)
    tmp = torch.diag_embed(a)
    t6 = time.time()

    R = V @ tmp @ U.transpose(2,3)

    r1cr = R @ r1c
    loss_all = torch.norm(r1cr * mask - r2c * mask, dim=(2, 3)) ** 2 / torch.norm(r2c * mask, dim=(2, 3)) ** 2
    loss_tr = torch.sum((loss_all) * batch_mask) / torch.sum(batch_mask)
    if return_coords:
        pred = r1cr[:,-1,:,:].squeeze().cpu().detach().numpy()
        target = r2c[:,-1,:,:].squeeze().cpu().detach().numpy()
        return loss_tr, pred, target
    else:
        return loss_tr