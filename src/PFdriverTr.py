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

dataFile = './../Data/testing'
id, seq, pssm2, entropy, dssp, RN, RCa, RCb, mask = pnetProcess.parse_pnet(dataFile)

idx = np.arange(0, 40)
ncourse = 4
X = []
Yobs = []
M = []
S = []
for i in idx:
    Xi, Yobsi, Mi, Shi, Si = pnetProcess.getProteinData(seq, pssm2, entropy, RN, RCa, RCb, mask, i, ncourse)
    X.append(Xi)
    ns = Si.shape[0]
    Yobs.append(Yobsi[0,0,:ns,:ns]/5000)
    M.append(Mi[0,:ns,:ns])
    S.append(Si.unsqueeze(0).type(torch.FloatTensor))

    print('Image ', i, 'imsize ', Xi.shape)

idx = 0

ntokens = 20 # the size of vocabulary
emsize  = 250 # embedding dimension
nhid    = 250 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead   = 2 # the number of heads in the multiheadattention models
dropout = 1e-6 #0.2 # the dropout value
ntokenOut = -1 # negative ntokenOut = ntoken

model   = networks.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout, ntokenOut) #.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ',total_params)
Z = S[idx] #[0,:,:]
Yp = model(Z)
Dp = networks.tr2DistSmall(Yp)
plt.imshow(Dp.detach())
plt.colorbar()


lr = 1e-3 # learning rate
#optimizer = optim.SGD(model.parameters(), lr=lr)
optimizer = optim.Adam(model.parameters(), lr=lr)
#scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.99)
iters = 1000

model.train() # Turn on the train mode
for itr in range(iters):
    #idx = torch.randint(0,40,(1,))
    data = S[idx] #[0,:,:]
    targets = Yobs[idx]
    Msk     = M[idx]
    optimizer.zero_grad()
    output = model(data)
    #output = networks.tr2Dist(output)
    output = networks.tr2DistSmall(output)

    loss = torch.norm(Msk*(output - targets))**2/torch.norm(Msk*targets)**2
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    print("% 2d, % 10.3E"% (itr, torch.sqrt(loss).item()))


Yp = model(S[idx])
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