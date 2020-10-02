from src import proTerp
from src import pnetProcess
from src import utils
from src import networks
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

## Get the data
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

X    = torch.load('../../../data/casp12Testing/Xtest.pt')
Yobs = torch.load('../../../data/casp12Testing/Coordtest.pt')
MSK  = torch.load('../../../data/casp12Testing/Masktest.pt')
S    = torch.load('../../../data/casp12Testing/Seqtest.pt')

pi = 3.1425926535
# define forward net
def get3DstructureCoo(dX,h=0.0651, sigma=1.0):
    n   = dX.shape[1]+1
    nrm = torch.sqrt(torch.sum(dX**2,dim=0,keepdim=True))
    dX  = dX/nrm*h
    X   = torch.zeros(3,n)
    for i in range(n-1):
        X[:,i+1] = X[:,i] + dX[:,i]

    D = torch.relu(torch.sum(X**2,axis=0,keepdim=True) + torch.sum(X**2,axis=0,keepdim=True).t() - 2*X.t()@X)
    Dout = torch.exp(-D / sigma)
    return Dout, X

def get3Dstructure(theta,psi,h=0.0651, sigma=1.0):

    n = len(theta)
    X = torch.zeros(3,n+1)
    dX = torch.zeros(3,n)
    dX[0,:] = h*torch.sin(theta)*torch.cos(psi)
    dX[1,:] = h*torch.sin(theta)*torch.sin(psi)
    dX[2,:] = h*torch.cos(theta)

    for i in range(n):
        X[:,i+1] = X[:,i] + dX[:,i]

    D    = torch.relu(torch.sum(X**2,axis=0, keepdim=True) + torch.sum(X**2,axis=0,keepdim=True).t() - 2*X.t()@X)
    Dout = torch.exp(-D/sigma)
    return Dout, X

def getPotentialData(D,S):
    # get the cov structure
    n  = S.shape[0]
    CS = torch.zeros(400,n,n)
    k  = 0
    for i in range(20):
        for j in range(20):
            CS[k,:,:] = 0.5*(torch.ger(S[:,i],S[:,j]) + torch.ger(S[:,j],S[:,i]))*D
            k += 1
    return CS

def getPotentialFunction(D,S,K):
    CS = getPotentialData(D, S)
    CS = CS.reshape((400,-1))
    l = len(K)
    Z = CS
    for i in range(l):
        Z = torch.relu(K[i]@Z)

    return torch.sum(Z)

def initialPotFun(sig=1e-2):
    K = nn.ParameterList([])
    Ki = nn.Parameter(sig*torch.randn(800,400))
    K.append(Ki)
    Ki = nn.Parameter(sig*torch.randn(1200,800))
    K.append(Ki)
    Ki = nn.Parameter(sig*torch.randn(300,1200))
    K.append(Ki)
    return K

def variationalNet(S,K,h,nt,dt,sigma=1):
    n     = S.shape[0]
    theta = nn.Parameter(1e-4*torch.randn(n-1))
    psi   = nn.Parameter(1e-4*torch.randn(n-1))
    optimizer = optim.SGD([{'params': theta},{'params': psi}], lr=dt, momentum=0.)
    for i in range(nt):
        optimizer.zero_grad()
        D, X = get3Dstructure(theta, psi,h,sigma)
        E = getPotentialFunction(D, S, K)
        dpsi   = torch.autograd.grad(E, psi, create_graph=True)[0]
        dtheta = torch.autograd.grad(E, theta, create_graph=True)[0]
        dpsi   = dpsi/torch.norm(dpsi)
        dtheta = dtheta /torch.norm(dtheta)

        psi   = psi   - dt*dpsi
        theta = theta - dt*dtheta

        print('         %d      %.3e '% (i, E.item()))

    return theta, psi, D, X

def variationalNetCoo(S,K,h,nt,dt,sigma=1):
    n     = S.shape[0]
    dX = nn.Parameter(1e-4*torch.randn(3,n-1))
    optimizer = optim.SGD([{'params': dX}], lr=dt, momentum=0.)
    for i in range(nt):
        optimizer.zero_grad()
        D, X = get3DstructureCoo(dX,h,sigma)
        E = getPotentialFunction(D, S, K)
        ddX   = torch.autograd.grad(E, dX, create_graph=True)[0]
        ndx = torch.norm(ddX)
        ddX   = ddX/torch.norm(ddX)

        dX   = dX   - dt*ddX
        print('         %d      %.3e       %.3e '% (i, E.item(), ndx.item()))

    return dX, D, X



if __name__ == "__main__":
    # run the distance
    Seq = S[0]
    ns  = Seq.shape[0]
    coo = Yobs[0][0,0,:,:ns]
    Do  = torch.relu(torch.sum(coo**2,0).unsqueeze(0) + torch.sum(coo**2, 0).unsqueeze(1) - 2*coo.t()@coo)
    Do  = torch.exp(-Do)
    M   = torch.ger(MSK[0][:ns],MSK[0][:ns])
    K = initialPotFun()
    numpar = sum(p.numel() for p in K)
    print('Number of network parameters ', numpar)
    nt = 5
    dt = 5e-2
    h  = 0.0651
    optimizerFit = optim.SGD([{'params': K}], lr=1e-2, momentum=0.)

    for i in range(20):
        optimizerFit.zero_grad()
        #theta, psi, D, X = variationalNet(Seq, K, h, nt, dt)
        dX, D, X = variationalNetCoo(Seq, K, h, nt, dt)

        lossFit = F.mse_loss(M*D,M*Do)/F.mse_loss(M*Do,0*Do)
        print('%d      %.3e ' % (i, lossFit.item()))
        lossFit.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(K, 0.5)
        optimizerFit.step()


    plt.imshow(D.detach())


# X -> D(X),S  -> C(D(X),S) (NN 1)
# E = f(C(D(X),S, Theta)   (NN 2)
# X^* = argmin f(C(D(X),S, Theta)
# min_{Theta} \|D(X^*(Theta)) -Dobs\|^2