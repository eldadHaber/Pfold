import numpy as np
import torch
import torch.nn as nn
from src import networks, optimizeNet, pnetProcess
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from src import utils
from src import regularization as rg

def imagesc(X):
    plt.imshow(X)
    plt.colorbar()

def plot(X):
    plt.plot(X)

Yobs = torch.load('../../../data/casp12Testing/Coordtest.pt')
MSK  = torch.load('../../../data/casp12Testing/Masktest.pt')
S     = torch.load('../../../data/casp12Testing/Seqtest.pt')

def getIterData(S,Yobs,MSK,i):
    M = MSK[i]
    X = Yobs[i][0, 0,:,:]
    m = X.shape[-1]
    Si = S[i].t()
    n  = Si.shape[1]
    Seq = torch.zeros(20,m)
    Seq[:,:n] = Si
    Seq[-1,n:] = 1

    #X = utils.linearInterp1D(X,M)
    #X = torch.tensor(X)
    D = torch.sum(torch.pow(X,2), dim=0, keepdim=True) + torch.sum(torch.pow(X,2), dim=0, keepdim=True).t() - 2*X.t()@X
    D = 0.5*(D+D.t())
    mm = torch.diag(M)
    D  = mm@D@mm

    U, Lam, V = torch.svd(D)

    C = torch.zeros(10,m)
    C[:5,:] = torch.diag(torch.sqrt(Lam[:5]))@U[:,:5].t()
    C[5:,:] = torch.diag(torch.sqrt(Lam[:5]))@V[:, :5].t()

    Ux, Lamx, Vx = torch.svd(X)

    return Seq, torch.diag(Lamx)@Vx.t(), M

n0 = 256
sc = 5
c  = 3
Arch = torch.tensor([[c,n0,1,sc],
                     [n0,n0,5,sc],
                     [n0,2*n0,1,sc],
                     [2*n0,2*n0,5,sc],
                     [2*n0,4*n0,1,sc],
                     [4*n0,4*n0,5,sc],
                     [4*n0,8*n0,1,sc]])
model = networks.vnet1D(Arch,20,h=1e-1)

#nstart = 3
#nopen = 32
#nhid  = 128
#nclose = 20
#nlayers = 256
#Ahyper = [nstart, nopen, nhid, nclose, nlayers]
#model = networks.hyperNet(Ahyper)


total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ',total_params)


lr = 1e-4
optimizer = optim.Adam([{'params': model.K, 'lr': lr},{'params': model.W, 'lr': lr}], lr=lr)
#optimizer = optim.Adam([{'params': model.Kopen, 'lr': 0.01},
#                        {'params': model.Kclose, 'lr': 0.01},
#                        {'params': model.W, 'lr': lr}], lr=lr)

alossBest = 1e6
ndata = len(S)-1
epochs = 500

#i=1
#Z, Coords, M = getIterData(S, Yobs, MSK, i)
#Zout = model(Coords.unsqueeze(0), M.unsqueeze(0))
#error

ndata = 2
bestModel = model
hist = torch.zeros(epochs)
for j in range(epochs):
    # Prepare the data
    aloss = 0
    for i in range(1,ndata):
    #for i in range(0, 1):

        Z, Coords, M = getIterData(S, Yobs, MSK, 0)
        optimizer.zero_grad()
        Zout = model(Coords.unsqueeze(0), M.unsqueeze(0))
        print(torch.norm(Zout).item())
        onehotZ = torch.argmax(Z, dim=0)
        loss = F.cross_entropy(Zout.squeeze(0).t(), onehotZ)
        #loss = torch.norm(M*Zout-M*Z)**2/torch.norm(M*Z)**2
        #loss.backward(retain_graph=True)
        loss.backward()

    #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)
        aloss += loss.detach()
        optimizer.step()
        normgW = model.W.grad.norm().item()
        #normgK = 0
        #for ii in range(len(model.K)):
        #    normgK += torch.norm(model.K[ii].grad)
        #normgK = normgK.item()
        print(i, '   ', j, '     ', loss.item(),'   ',normgW)  #,'   ',normgK)
    if aloss < alossBest:
        alossBest = aloss
        bestModel = model

    # Validation on 0-th data
    with torch.no_grad():
        Z, Coords, M = getIterData(S, Yobs, MSK, 0)
        onehotZ = torch.argmax(Z, dim=0)
        Zout = model(Coords.unsqueeze(0), M.unsqueeze(0))
        #lossV = torch.norm(M*Zout-M*Z)**2/torch.norm(M*Z)**2
        lossV = F.cross_entropy(Zout.squeeze(0).t(), onehotZ)
    print('==== Epoch =======',j, '        ', (aloss).item()/(ndata-1),'   ',lossV.item())
    hist[j] = (aloss).item()/(ndata)

