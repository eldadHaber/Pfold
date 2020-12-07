import numpy as np
import torch
import torch.nn as nn
from src import networks, optimizeNet, pnetProcess
import matplotlib.pyplot as plt
import torch.optim as optim
from src import utils
from src import regularization as rg



def imagesc(X):
    plt.imshow(X)
    plt.colorbar()

def plot(X):
    plt.plot(X)

dataFile = '../../../data/casp12Testing/testing.pnet'
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

n0 = 128
sc = 3
Arch = torch.tensor([[20,n0,1,sc],[n0,n0,5,sc],[n0,2*n0,1,sc],[2*n0,2*n0,5,sc],[2*n0,4*n0,1,sc],[4*n0,4*n0,5,sc],[4*n0,8*n0,1,sc]])
model = networks.vnet1D(Arch,3)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ',total_params)


lr = 1e-3 # learning rate
#optimizer = optim.SGD(model.parameters(), lr=lr)
optimizer = optim.Adam([{'params': model.K, 'lr': lr},{'params': model.W, 'lr': lr}], lr=lr)
#scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.99)
iters = 50
hist = []
model.train() # Turn on the train mode
for itr in range(iters):
    idx = 0 #torch.randint(4,5,(1,))
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
    Ypred = model(data, Msk[:,0])
    output = Ypred.squeeze(0).t().unsqueeze(0)
    output = utils.tr2Dist(output)

    Wt = 1 /(targets + 1e-3)
    misfit = torch.norm(Wt*(Msk*(output - targets)))**2/torch.norm(Wt*(Msk*targets))**2

    dY  = Ypred[0,:,1:] - Ypred[0,:,:-1]
    reg = torch.sum(torch.sqrt(torch.sum(dY**2,dim=1)))
    loss = misfit + 1e-5*reg
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    if len(hist) > 40:
        aloss = torch.mean(torch.tensor(hist[-40:])).item()
    else:
        aloss = torch.sqrt(misfit).item()
    print("% 2d  % 10.3E   % 10.3E   % 10.3E"% (itr, torch.sqrt(misfit).item(),aloss, reg))
    hist.append(torch.sqrt(misfit).item())


plt.figure(2)
plt.subplot(2,2,1)
imagesc(Msk*output.detach())
plt.subplot(2,2,2)
imagesc(Msk*targets)
plt.subplot(2,2,3)
imagesc(torch.abs(Msk*(targets-output.detach())))

