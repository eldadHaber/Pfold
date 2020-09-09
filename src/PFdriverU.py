import numpy as np
import torch
import torch.nn as nn
from src import networks, optimizeNet, pnetProcess
import matplotlib.pyplot as plt
import torch.optim as optim


def imagesc(X):
    plt.imshow(X)
    plt.colorbar()

def plot(X):
    plt.plot(X)

dataFile = './../data/testing'
id, seq, pssm2, entropy, dssp, RN, RCa, RCb, mask = pnetProcess.parse_pnet(dataFile)

idx = np.arange(0, 40)
ncourse = 4
X = []
Yobs = []
M = []
S = []
rM = []
for i in idx:
    Xi, Yobsi, Mi, Shi, Si, rMi = pnetProcess.getProteinData(seq, pssm2, entropy, RN, RCa, RCb, mask, i, ncourse)
    X.append(Xi)
    ns = Si.shape[0]
    Yobs.append(Yobsi[0,0,:ns,:ns]/5000)
    M.append(Mi[0,:ns,:ns])
    S.append(Si.unsqueeze(0).type(torch.FloatTensor))
    rM.append(rMi.type(torch.FloatTensor))
    print('Image ', i, 'imsize ', Xi.shape)


Arch = torch.tensor([[20,64,1,5],[64,64,5,5],[64,128,1,5],[128,128,15,5],[128,256,1,5]])
model = networks.vnet1D(Arch,32)

#id = 0
#Z = S[id].squeeze(0).t()
#n = 8*(Z.shape[1]//8 + 1)
#Zp = torch.zeros(1,Z.shape[0],n)
#Zp[:,:,:Z.shape[1]] = Z
#Yp = model(Zp)
#Dp = networks.tr2DistSmall(Yp.squeeze(0).t().unsqueeze(0))
#plt.imshow(Dp.detach())
#plt.colorbar()

lr = 1e-3 # learning rate
#optimizer = optim.SGD(model.parameters(), lr=lr)
optimizer = optim.Adam([{'params': model.K, 'lr': lr},{'params': model.W, 'lr': lr}], lr=lr)
#scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.99)
iters = 200
hist = []
model.train() # Turn on the train mode
for itr in range(iters):
    idx = torch.randint(0,40,(1,))
    # Arrange the data for a convolution
    Z  = S[idx].transpose(1,2)
    no = Z.shape[2]
    n  = 8 * (no // 8 + 1)
    data = torch.zeros(1, Z.shape[1], n)
    data[:, :, :no] = Z
    targets = torch.zeros(n, n)
    targets[:no,:no] = Yobs[idx]
    Msk     = torch.zeros(n, n)
    Msk[:no,:no] = M[idx]
    optimizer.zero_grad()
    Ypred = model(data)
    output = Ypred.squeeze(0).t().unsqueeze(0)
    output = networks.tr2DistSmall(output)

    Wt = 1 /(targets + 1e-3)

    loss = torch.norm(Wt*(Msk*(output - targets)))**2/torch.norm(Wt*(Msk*targets))**2
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    print("% 2d, % 10.3E"% (itr, torch.sqrt(loss).item()))
    hist.append(torch.sqrt(loss).item())


plt.figure(2)
plt.subplot(2,2,1)
imagesc(Msk*output.detach())
plt.subplot(2,2,2)
imagesc(Msk*targets)
plt.subplot(2,2,3)
imagesc(torch.abs(Msk*(targets-output.detach())))

