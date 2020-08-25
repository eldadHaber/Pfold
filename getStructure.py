import numpy as np
import torch

def dist(X,mask=1.0):
    D =  torch.relu(torch.sum(X**2,dim=1).unsqueeze(1) + torch.sum(X**2,dim=1).unsqueeze(0) - 2*X@X.t())
    return mask*D

def JdX(X,dX,M=1.0):
    V = 2*M*(torch.sum(X*dX,axis=1).unsqueeze(1) + torch.sum(X*dX,axis=1).unsqueeze(0) - X@dX.t() - dX@X.t())
    return V

def JTdV(X,dV,M=1.0):
    n1 = X.shape[0]
    e2 = torch.ones(3,1,  dtype=X.dtype)
    e1 = torch.ones(n1,1, dtype=X.dtype)
    E12 = e1@e2.t()
    dV  = M*dV
    dX = 2*X*(dV@E12)  + 2*(X*(dV.t()@E12)) - 2*dV.t()@X - 2*dV@X

    return dX


def cglsMF(B, X, M=1.0, tol=1e-2, maxIter=100):
# Matrix Free CGLS

    R = B
    D = JTdV(X, R, M)
    W = torch.zeros(D.shape, dtype=D.dtype)
    normr2 = torch.norm(D)**2

    for j in range(maxIter):
        Ad    = JdX(X, D, M)
        nrmAd2 = torch.norm(Ad)**2
        alpha = normr2 / nrmAd2
        W = W + alpha*D
        R = R - alpha*Ad
        S = JTdV(X, R, M)
        normr2New = torch.norm(S)**2
        if normr2New < tol:
            return W

        beta = normr2New / normr2
        normr2 = normr2New
        D = S + beta*D

        #print(j  , (torch.norm(R)/torch.norm(B)).item(), torch.norm(W).item())

    return W


def getXfromD(Coords, DistMap, M=1.0, niter=100, tol=1e-3):

    for i in range(niter):
        DistMapC = dist(Coords, M)
        ResMap   = DistMapC - DistMap
        F  = 0.5 * torch.norm(ResMap)**2
        dF = JTdV(Coords, ResMap, M)
        S  = cglsMF(ResMap, Coords, M=M, tol=1e-2, maxIter=100)

        print("% 2d, % 2d,   % 10.3E,   % 10.3E" % (i, 0, F.item(), torch.norm(dF).item()))
        muls = 1e0
        itc = 1
        while 1:
            Coordst = Coords - muls * S
            ResMapt = dist(Coordst,M) - DistMap
            Ft = 0.5 * torch.norm(ResMapt)**2
            print("% 2d, % 2d,   % 10.3E" % (i, itc, Ft.item()))
            if Ft < F:
                break

            muls = muls/2
            itc = itc + 1
            if itc > 5:
                print('LSB')
                return Coords, DistMapC

        Coords = Coordst
        if torch.norm(ResMap) < tol:
            print('Converge')
            return Coords, DistMapC

    return Coords, DistMapC