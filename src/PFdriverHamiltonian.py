import numpy as np
import torch
import torch.nn as nn
from src import networks, optimizeNet, pnetProcess
import matplotlib.pyplot as plt
import torch.optim as optim

dataFile = './../data/testing'
id, seq, pssm2, entropy, dssp, RN, RCa, RCb, mask = pnetProcess.parse_pnet(dataFile)

idx = np.arange(0, 1)
ncourse = 4
X = []
Yobs = []
M = []
Sh = []
S = []
for i in idx:
    Xi, Yobsi, Mi, Shi, Si = pnetProcess.getProteinData(seq, pssm2, entropy, RN, RCa, RCb, mask, i, ncourse)
    X.append(Xi)
    Yobs.append(Yobsi)
    M.append(Mi)
    Sh.append(Shi)
    S.append(Si)
    print('Image ', i, 'imsize ', Xi.shape)

S   = S[0].t().unsqueeze(0)
S = S.type(torch.FloatTensor)
Yobs = Yobs[0]
M    = M[0]


no = 256
nh = 512
A = torch.tensor([[no,   20,  1],
                  [no,  nh,   18],
                  [25,   no,   1]])

K, Wopen, Wclose = networks.inithNetParams(A)
h = 0.1
net = networks.hNet(K, Wopen, Wclose, h)

Y = net(S)
D = networks.Seq2Dist(Y)
lr = [1e-2, 5e-3, 5e-3]
iters = 1000
optimizer = optim.Adam([{'params': net.K, 'lr': lr[0]},{'params': net.Wopen, 'lr': lr[1]},
                        {'params': net.Wclose, 'lr': lr[2]}])

runningLoss = 0.0
hist = []
cnt  = 0
for itr in range(iters):
    indx = torch.randint(0, len(X),[1])
    Xi = S    #X[indx]
    ns     = Xi.shape[2]
    Yi = Yobs[0,0,:ns,:ns] #Y[indx]
    Mi = M[0,:ns,:ns]    #M[indx]
    # Forward pass
    optimizer.zero_grad()

    Ypred  = net(Xi)
    Ypred  = networks.Seq2Dist(Ypred)
    loss   = torch.norm(Mi*(Ypred-Yi))**2/torch.norm(Mi*Yi)**2

    hist.append(loss.item())
    loss.backward()
    optimizer.step()

    runningLoss += loss.item()
    cnt         += 1
    # Print progress
    nprnt = 1
    if itr%nprnt == 0:
        dK   = net.K.grad
        dWo  = net.Wopen.grad
        dWc  = net.Wclose.grad

        nk   = torch.max(torch.abs(dK))
        nwo  = torch.max(torch.abs(dWo))
        nwc  = torch.max(torch.abs(dWc))

        # Iterations  Running loss    loss      misfit    reg   nK  nW
        print("% 2d, % 10.3E,   % 10.3E,  % 10.3E, % 10.3E, % 10.3E"
              % (itr, runningLoss/cnt, loss.item(),nk.item(), nwo.item(), nwc.item()))
        if nk+nwo + nwc < 1e-5:
            print('Converge 0')
            break
        if runningLoss/cnt < 1e-3:
            print('Converge ')
            break
        runningLoss = 0.0
        cnt = 0.0


Yp = net(S)
Yp = networks.Seq2Dist(Yp)
plt.imshow(M[0,:ns,:ns]*Yp.detach())