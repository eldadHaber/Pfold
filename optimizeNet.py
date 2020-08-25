import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import networks


def trainNetwork(dnn, X, Y, M, iters, lr, regpar = 1e-4, dweights = torch.tensor([1.0,1,1,1]), stopTol = 5e-3):
    """
    Train network with given parameters using our semi-symmetric loss.
    """
    optimizer = optim.Adam([{'params': dnn.K, 'lr': lr[0]},{'params': dnn.W, 'lr': lr[1]}])
    #
    runningLoss = 0.0
    hist = []
    cnt  = 0
    for itr in range(iters):
        indx = torch.randint(0, len(X),[1])
        Xi = X[indx]
        Yi = Y[indx]

        # Forward pass
        optimizer.zero_grad()

        Ypred  = dnn(Xi)
        misfit = networks.misfitFun(Ypred, Yi, M, dweights)
        reg, normGrad    = networks.TVreg(Ypred, M)

        # Calculate loss and backpropagate
        loss = misfit + regpar*reg
        hist.append([loss.item(), misfit.item(), reg.item()])
        loss.backward()
        optimizer.step()

        runningLoss += loss.item()
        cnt         += 1
        # Print progress
        nprnt = 1
        if itr%nprnt == 0:
            dK  = getGrad(dnn.K)
            dW  = dnn.W.grad

            nk  = getMax(dK)
            nw  = torch.max(torch.abs(dW))

            # Iterations  Running loss    loss      misfit    reg   nK  nW
            print("% 2d, % 10.3E,   % 10.3E,  % 10.3E, % 10.3E , % 10.3E, % 10.3E"
                  % (itr, runningLoss/cnt, loss.item(), torch.sqrt(misfit).item(),
                     reg.item(), nk.item(), nw.item()))
            if nk+nw < 1e-5:
                print('Converge 0')
                return dnn, np.asarray(hist)
            if runningLoss/cnt < stopTol:
                print('Converge ')
                return dnn, np.asarray(hist)
            runningLoss = 0.0
            cnt = 0.0

    return dnn, np.asarray(hist)


def getGrad(K):
    gradK = []
    for i in range(len(K)):
        gradK.append(K[i].grad)

    return gradK

def getMax(dK):
    mx = -999999.0
    for i in range(len(dK)):
        mxt = torch.max(torch.abs(dK[i]))
        if mxt > mx:
            mx = mxt

    return mx

def getCoordsFromDist(X,D,lr=1e-2,niter=100):

    optimizer = optim.Adam([{'params': X, 'lr': lr}])
    for i in range(niter):
        optimizer.zero_grad()
        Dpred = torch.max(torch.sum(X**2,1).unsqueeze(1) + torch.sum(X**2,1).unsqueeze(0) - 2*X@X.t())
        loss  = 0.5*torch.sum((D-Dpred)**2)
        loss.backward()
        optimizer.step()

        dX = X.grad
        nx = torch.max(torch.abs(dX))

        # Iterations  loss    nX
        print("% 2d, % 10.3E,   % 10.3E" % (i, loss.item(),  nx.item()))
        if nx < 1e-5:
            print('Converge 0')
            return X
        if loss  < 1e-3:
            print('Converge ')
            return X

    return X