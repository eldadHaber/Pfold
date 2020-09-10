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

dataFile = './../data/testing.pnet'
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


ntokens = 20 # the size of vocabulary
emsize  = 250 # embedding dimension
nhid    = 250 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 8 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead   = 10 # the number of heads in the multiheadattention models
dropout = 1e-6 #0.2 # the dropout value
ntokenOut = 3 # negative ntokenOut = ntoken
stencil   = 5
model   = networks.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout, ntokenOut, stencil) #.to(device)

#model   = tr.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout, ntokenOut, stencil) #.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ',total_params)

id = 0
Z = S[id].squeeze(0).unsqueeze(1) #[0,:,:]
Yp = model(Z)
Yp = Yp.squeeze(1).unsqueeze(0)
#Dp = torch.relu(torch.sum(Yp**2,2) + torch.sum(Yp**2,2).t() - 2*Yp.squeeze(1)@Yp.squeeze(1).t())
Dp = networks.tr2DistSmall(Yp)
#plt.imshow(Dp.detach())
#plt.colorbar()

#error

lr = 1e-4 # learning rate
#optimizer = optim.SGD(model.parameters(), lr=lr)
optimizer = optim.Adam(model.parameters(), lr=lr)
#scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.99)
iters = 200
hist = []
ids = [0]
model.train() # Turn on the train mode
for itr in range(iters):
    idx = ids[0] #ids[torch.randint(0,8,(1,))]
    data = S[idx].squeeze(0).unsqueeze(1) #[0,:,:]
    targets = Yobs[idx]
    Msk     = M[idx]
    optimizer.zero_grad()
    output = model(data)
    output = output.squeeze(1).unsqueeze(0)
    #output = networks.tr2Dist(output)
    output = networks.tr2DistSmall(output)
    #output = torch.sqrt(torch.relu(torch.sum(output ** 2, 2) + torch.sum(output ** 2, 2).t() - 2 *
    #                   output.squeeze(1) @ output.squeeze(1).t()))

    #Wt  = 1/torch.sqrt(targets + 0.01)
    Wt = 1 /(targets + 1e-3)
    misfit = torch.norm(Wt*(Msk*(output - targets)))**2/torch.norm(Wt*(Msk*targets))**2
    tv, tt = networks.TVreg(output.unsqueeze(0).unsqueeze(0))
    loss = misfit + 1e-5*tv
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    print("% 2d  % 10.3E  % 10.3E"% (itr, torch.sqrt(misfit).item(), tv.item()))
    hist.append(torch.sqrt(loss).item())

idx = 0
Yp = model(S[idx].squeeze(0).unsqueeze(1))
Yp = Yp.squeeze(1).unsqueeze(0)
Dp = networks.tr2DistSmall(Yp)
plt.subplot(2,2,1)
plt.imshow(M[idx]*Dp.detach())
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(Yobs[idx])
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(torch.abs(M[idx]*Dp.detach() - Yobs[idx]))
plt.colorbar()
print("Done")
plt.subplot(2,2,4)
plt.imshow(Dp.detach())
plt.colorbar()

r = M[idx]*(Dp.detach()-Yobs[idx])
print('verror  ',torch.norm(r)/torch.norm(M[idx]*Yobs[idx]))