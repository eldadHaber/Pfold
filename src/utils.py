import numpy as np
import scipy
# import scipy.spatial
import string
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class list2np(object):
    def __init__(self):
        pass
    def __call__(self, *args):
        args_array = ()
        for arg in args:
            args_array += (np.asarray(arg),)
        return args_array

    def __repr__(self):
        return self.__class__.__name__ + '()'

def getPointDistance(output, targets, alpha=0.5):

    #outRot, tarRot, R = rotatePoints(output.squeeze(0), targets.squeeze(0))
    #doutRot = outRot[:, 1:] - outRot[:, :-1]
    #dtarRot = tarRot[:, 1:] - tarRot[:, :-1]
    misfitDis = 0.0
    for i in range(output.shape[0]):
        outi  = output[i,:,:]
        tari = targets[i,:,:]
        Dc = torch.sum(outi**2, dim=0, keepdim=True) + torch.sum(outi**2, dim=0,
                                                                 keepdim=True).t() - 2*outi.t()@outi
        Dc = torch.sqrt(torch.relu(Dc))
        Do = torch.sum(tari**2, dim=0, keepdim=True) + torch.sum(tari**2, dim=0,
                                                                  keepdim=True).t() - 2*tari.t()@tari
        Do = torch.sqrt(torch.relu(Do))
        misfitDis += F.mse_loss(Dc, Do) / F.mse_loss(Do, 0*Do)
    misfitDis = misfitDis/output.shape[0]

    #misfitCoo = F.mse_loss(doutRot, dtarRot) / F.mse_loss(dtarRot, dtarRot * 0)
    misfitCoo = F.mse_loss(output, targets) / F.mse_loss(targets, targets * 0)

    misfit = alpha*misfitDis + (1-alpha)*misfitCoo
    return misfit, misfitDis, misfitCoo

def getRandomMask(n,m):
    mask = torch.ones(m)
    i = torch.randint(0,m-n,(1,))
    mask[i:i+n] = 0
    return mask


def getRandomCrop(X,M, winsize=64, batchSize=[]):

    # Find potential windows
    k = X.shape[-1]
    ind = []
    T = []
    for i in range(k-winsize):
        t = torch.sum(M[i:i+winsize])
        if t == winsize:
            ind.append(i)
            T.append(X[:,:,i:i+winsize])

    # Choose on in random
    n = len(ind)
    Xout = []
    Tout = []
    if n > 0:
        ii = torch.randint(0,n,(1, ))
        Xout = X[:,:,ind[ii]:ind[ii]+winsize]
        Tout = torch.Tensor(len(T),X.shape[1],winsize)
        for i in range(n):
            Tout[i,:,:] = T[i]
    if len(batchSize)>0:
        if n>batchSize[0]:
            jj = torch.randint(0,n-batchSize[0],(1,))
            Tout = Tout[jj:jj+batchSize[0],:,:]
    return Xout, Tout


def getRotMat(t):
    A1 = torch.tensor([[torch.cos(t[0]), -torch.sin(t[0]), 0],
                       [torch.sin(t[0]), torch.cos(t[0]), 0],
                         [0, 0, 1.0]])
    A2 = torch.tensor([[torch.cos(t[1]),  0, -torch.sin(t[1])],
                       [0, 1.0, 0],
                       [torch.sin(t[1]), 0 , torch.cos(t[1])]])
    A3 = torch.tensor([[1,0,0],
                       [0, torch.cos(t[2]), -torch.sin(t[2])],
                       [0, torch.sin(t[2]), torch.cos(t[2])]])
    return A1@A2@A3

def tr2Dist(Y):

    k = Y.shape[2]
    Z = Y[0,:,:]
    Z = Z - torch.mean(Z, dim=0, keepdim=True)
    D = torch.sum(Z**2, dim=1).unsqueeze(0) + torch.sum(Z**2, dim=1).unsqueeze(1) - 2*Z@Z.t()
    D = 3*D/k
    return torch.sqrt(torch.relu(D))

def Seq2Dist(Y):

    D = torch.sum(Y**2,dim=1) + torch.sum(Y**2,dim=1).t() -  Y[0,:,:].t()@Y[0,:,:]
    return D

def rotatePoints(X, Xo):
    # Find a matrix R such that Xo@R = X
    # Translate X to fit Xo
    # (X+c)R = V*S*V' R + c*R = Xo
    # X = Uo*So*Vo'*R' - C

    # R = V*inv(S)*U'
    X = X.squeeze(0).t()
    Xo = Xo.t()
    if X.shape != Xo.shape:
        U, S, V =  torch.svd(X)
        X = U[:,:3]@torch.diag(S[:3])@V[:3,:3].t()
        #S[3:] = 0
        #X = U@torch.diag(S)@V.t()
        #X = X[:,:3]

    n, dim = X.shape

    Xc  = X - X.mean(dim=0)
    Xco = Xo - Xo.mean(dim=0)

    C = (Xc.t()@Xco) / n

    U, S, V = torch.svd(C)
    d = torch.sign((torch.det(U) * torch.det(V)))

    R  = V@torch.diag(torch.tensor([1.0,1,d],dtype=U.dtype))@U.t()

    Xr = Xc@R.t()
    #print(torch.norm(Xco - Xc @ R.t()))

    return Xr.t(), Xco.t(), R

def getRotDist(Xc, Xo, alpha = 1.0):

    Xr, Xco, R = rotatePoints(Xc, Xo)
    Do = torch.sum(Xo**2,dim=1,keepdim=True) + torch.sum(Xo**2,dim=1,keepdim=True).t() - 2*Xo@Xo.t()
    Do = torch.sqrt(torch.relu(Do))
    Dc = torch.sum(Xc**2,dim=1,keepdim=True) + torch.sum(Xc**2,dim=1,keepdim=True).t() - 2*Xc@Xc.t()
    Dc = torch.sqrt(torch.relu(Dc))

    return F.mse_loss(Xr, Xco) + alpha*F.mse_loss(Dc, Do)
# Define some functions

def move_tuple_to(args,device,non_blocking=True):
    new_args = ()
    for arg in args:
        new_args += (arg.to(device,non_blocking=non_blocking),)
    return new_args


def fix_seed(seed, include_cuda=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # if you are using GPU
    if include_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def determine_network_param(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)



def getDistMat(X,msk=torch.tensor([1.0])):
    D = torch.sum(torch.pow(X,2), dim=0, keepdim=True) + torch.sum(torch.pow(X,2), dim=0, keepdim=True).t() - 2*X.t()@X
    mm = torch.ger(msk,msk)
    return mm*torch.sqrt(torch.relu(D))

def getNormMat(N,msk=torch.tensor([1.0])):
    N = N/torch.sqrt(torch.sum(N**2,dim=0,keepdim=True)+1e-9)
    D = N.t()@N
    mm = torch.ger(msk, msk)
    return mm*D

def orgProtData(x,normals,s, msk, sigma=1.0):
    n = s.shape[1]
    D = getDistMat(x,msk)
    D = torch.exp(-sigma*D)
    N = getNormMat(normals,msk)
    XX = torch.zeros(20, 20, n, n)
    NN = torch.zeros(20, 20, n, n)
    mm = torch.ger(msk, msk)
    mm = mm.view(-1)

    for i in range(20):
        for j in range(20):
            sij = 0.5*(torch.ger(s[i, :], s[j, :]) + torch.ger(s[j, :], s[i, :]))
            XX[i, j, :, :] = sij * D
            NN[i, j, :, :] = sij * N

    XX = XX.reshape((400, -1))
    NN = NN.reshape((400, -1))
    # XX = XX[:, mm > 0]
    # NN = NN[:, mm > 0]
    return XX, NN

def getGraphLap(X,sig=0.2):

    # normalize the data
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X/torch.sqrt(torch.sum(X**2,dim=1,keepdim=True)/X.shape[1] + 1e-4)
    # add  position vector
    pos = 0.5*torch.linspace(0, 1, X.shape[1]).unsqueeze(0)
    Xa = torch.cat((X, pos), dim=0)
    W = getDistMat(Xa)
    W = torch.exp(-W/sig)
    D = torch.diag(torch.sum(W, dim=0))
    L = D - W
    Dh = torch.diag(1/torch.sqrt(torch.diag(D)));
    L = Dh * L * Dh

    L = 0.5 * (L + L.t())

    return L, W

def randsvd(A,k):

    n = A.shape
    Omega = torch.randn(n[1],k,dtype=A.dtype)
    Y     = A@Omega
    Q,R   = torch.qr(Y)
    B     = Q.t()@A
    U, S, V = torch.svd(B)
    U = Q@U
    return U, S, V

def cheby(K,L,Z):
# apply T_k(x) = 2x*T_{k-1}(x) - T_{k-2}(x)
# KZ = \sum K[i] T_i(Z)
    n  = len(K)
    T0 = torch.ones(Z.shape) #torch.eye(L.shape[0])
    A  = K[0]*T0 #K[0]*Z@T0
    T1 = Z@L #L
    A  = A + K[1]*T1 #A + K[1]*Z@T1
    for i in range(2,n):
        T2 = 2*T1@L - T0
        A  = A + K[i]*T2
        T0 = T1.clone()
        T1 = T2.clone()
    return A
