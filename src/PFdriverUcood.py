import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
from src import networks, optimizeNet, pnetProcess
import matplotlib.pyplot as plt
import torch.optim as optim
from src import utils
from src import regularization as rg
import torch.nn.functional as F

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
MSK = []
S = []
rM = []
for i in idx:
    Xi, Yobsi, Mi = pnetProcess.getProteinDataLinear(seq, pssm2, entropy, RN, RCa, RCb, mask, i, ncourse)
    X.append(Xi)
    Yobs.append(Yobsi/5000)
    MSK.append(Mi)
    print('Image ', i, 'imsize ', Xi.shape)

n0 = 128
sc = 3
Arch = torch.tensor([[21,n0,1,sc],
                     [n0,n0,5,sc],
                     [n0,2*n0,1,sc],
                     [2*n0,2*n0,5,sc],
                     [2*n0,4*n0,1,sc],
                     [4*n0,4*n0,5,sc],
                     [4*n0,8*n0,1,sc]])
model = networks.vnet1D(Arch,3)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ',total_params)

Xp = model(X[0],MSK[0])
#print(Xp.shape)


lr = 1e-3 # learning rate
#optimizer = optim.SGD(model.parameters(), lr=lr)
optimizer = optim.Adam([{'params': model.K, 'lr': lr},{'params': model.W, 'lr': lr}], lr=lr)
#scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.99)
iters = 50000
hist = []
model.train() # Turn on the train mode
for itr in range(iters):
    idx = torch.randint(1,40,(1,))
    # Arrange the data for a convolution
    data    = X[idx]
    targets = Yobs[idx][0,0,:,:]

    Msk     = MSK[idx]
    optimizer.zero_grad()
    output = model(data,Msk)

    outRot, tarRot, R = utils.rotatePoints(output, targets)

    doutRot  = outRot[:, 1:] - outRot[:, :-1]
    dtarRot  = tarRot[:, 1:] - tarRot[:, :-1]
    output = output.squeeze(0)
    Dc = torch.sum(output**2,dim=0,keepdim=True) + torch.sum(output**2,dim=0,keepdim=True).t() - 2*output.t()@output
    Dc = torch.sqrt(torch.relu(Dc))
    Do = torch.sum(targets**2,dim=0,keepdim=True) + torch.sum(targets**2,dim=0,keepdim=True).t() - 2*targets.t()@targets
    Do = torch.sqrt(torch.relu(Do))
    M  = Msk.unsqueeze(1)@Msk.unsqueeze(0)
    mp = Msk[1:]  != 0
    mm = Msk[:-1] != 0
    dMsk = 1.0*torch.logical_and(mp, mm)
    misfitDis = F.mse_loss(M*Dc,M*Do)/F.mse_loss(M*Do,0*Do)
    misfitCoo = F.mse_loss(dMsk * doutRot, dMsk * dtarRot) / F.mse_loss(dMsk * dtarRot, dtarRot * 0)
    misfit = 5*misfitDis + misfitCoo

    reg = torch.sum(torch.sqrt(torch.sum(doutRot ** 2, dim=1)))
    loss = misfit + 1e-9*reg
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    if len(hist) > 40:
        aloss = torch.mean(torch.tensor(hist[-40:])).item()
    else:
        aloss = torch.sqrt(misfit).item()
    print("% 2d% 2d  % 10.3E   % 10.3E   % 10.3E   % 10.3E"% (itr, idx,  torch.sqrt(misfitDis).item(), torch.sqrt(misfitCoo).item(), aloss, reg))
    hist.append(torch.sqrt(misfit).item())


####### Attempt validation
idx = 1
data = X[idx]
targets = Yobs[idx][0, 0, :, :]
Msk = MSK[idx]
optimizer.zero_grad()
output = model(data, Msk)
outRot, tarRot, R = utils.rotatePoints(output, targets)

doutRot = outRot[:, 1:] - outRot[:, :-1]
dtarRot = tarRot[:, 1:] - tarRot[:, :-1]
output = output.squeeze(0)
Dc = torch.sum(output ** 2, dim=0, keepdim=True) + torch.sum(output ** 2, dim=0,
                                                             keepdim=True).t() - 2 * output.t() @ output
Dc = torch.sqrt(torch.relu(Dc))
Do = torch.sum(targets ** 2, dim=0, keepdim=True) + torch.sum(targets ** 2, dim=0,
                                                              keepdim=True).t() - 2 * targets.t() @ targets
Do = torch.sqrt(torch.relu(Do))
M = Msk.unsqueeze(1) @ Msk.unsqueeze(0)
mp = Msk[1:] != 0
mm = Msk[:-1] != 0
dMsk = 1.0 * torch.logical_and(mp, mm)
misfitDis = F.mse_loss(M * Dc, M * Do) / F.mse_loss(M * Do, 0 * Do)
misfitCoo = F.mse_loss(dMsk * doutRot, dMsk * dtarRot) / F.mse_loss(dMsk * dtarRot, dtarRot * 0)
misfit = misfitDis + misfitCoo


print("% 2d  % 10.3E   % 10.3E   % 10.3E   % 10.3E" % (
itr, torch.sqrt(misfitDis).item(), torch.sqrt(misfitCoo).item(), aloss, reg))

outRot = outRot.detach()
Dc    = Dc.detach()
plt.figure(1)
plt.subplot(2,2,1)
imagesc(M*Dc.detach())
plt.subplot(2,2,2)
imagesc(M*Do)
plt.subplot(2,2,3)
imagesc(torch.abs(M*Dc.detach() - M*Do))

plt.figure(2)
plt.subplot(3,1,1)
plt.plot(outRot[0, :])
plt.plot(tarRot[0, :])
plt.subplot(3,1,2)
plt.plot(outRot[1, :])
plt.plot(tarRot[1, :])
plt.subplot(3,1,3)
plt.plot(outRot[2, :])
plt.plot(tarRot[2, :])

fig = plt.figure(3)
ax = fig.gca(projection='3d')
ax.plot(Msk * outRot[0, :], Msk * outRot[1, :], Msk * outRot[2, :], label='parametric curve')
ax.plot(Msk * tarRot[0, :], Msk * tarRot[1, :], Msk * tarRot[2, :], label='parametric curve')
