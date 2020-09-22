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
dataFile = '../../../data/casp12Testing/testing.pnet'
id, seq, pssm2, entropy, dssp, RN, RCa, RCb, mask = pnetProcess.parse_pnet(dataFile)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
idx = np.arange(0, 40)
ncourse = 4
X = []
Yobs = []
MSK = []
S = []
rM = []
for i in idx:
    Xi, Yobsi, Mi = pnetProcess.getProteinDataLinear(seq, pssm2, entropy, RN, RCa, RCb, mask, i, ncourse,inter=False)
    X.append(Xi)
    Yobs.append(Yobsi/5000)
    MSK.append(Mi)
    print('Image ', i, 'imsize ', Xi.shape)


n0 = 128
sc = 3
Ag = torch.tensor([[24,n0,1,sc],
                    [n0,n0,5,sc],
                    [n0,2*n0,1,sc],
                    [2*n0,2*n0,5,sc],
                    [2*n0,4*n0,1,sc],
                    [4*n0,4*n0,5,sc],
                    [4*n0,8*n0,1,sc]])
#Ad = Ag
# Call generative and discreminative networks
#netG = proTerp.proGen(Ag,3)
netG = proTerp.vnet1D(Ag,3).to(device)
#lrD   = 1e-4
lrG   = 1e-3
beta1 = 0.5
#optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))
epochs     = 300
winsize    = 128

numdat = 10
ehist = torch.zeros(epochs)
hist = torch.zeros(numdat)
for epoch in range(epochs):
    for i in range(numdat):
        # Get a protein
        flag = True
        jj = i
        while flag:
            Xi = X[jj]
            Yi = Yobs[jj][:,0,:,:]
            Mi = MSK[jj]
            input = torch.zeros(1, 24, Xi.shape[2])
            input[:, :21, :] = Xi
            input[:, 21:, :] = Yi
            # try to obtain in-out patches
            patchIn, patches  = utils.getRandomCrop(input, Mi, winsize=winsize, batchSize=[32])
            if len(patchIn) > 0:
                #patchOut =  patchIn[:, 21:, :]
                patchOut = patches[:, 21:, :]
                flag = False
            else:
                jj += 1
        # Now train the generator
        maskSize = torch.randint(16,32,(1,)).item()
        randomMask = utils.getRandomMask(maskSize, winsize)
        mm = torch.ones(24, 128); mm[21:, :] = randomMask

        netG.zero_grad()
        #patchOutFake = netG(randomMask*patchIn, randomMask)
        patchesIn = (mm * patches).to(device)
        randomMask = randomMask.to(device)
        patchOutFake = netG(patchesIn, randomMask)
        #patchOutFake = randomMask*patchOut + (1-randomMask)*patchOutFake
        #errG = F.mse_loss(patchOutFake, patchOut)
        alpha = (0.98 ** (epoch / epochs * 300))
        errG, errGD, errGC = utils.getPointDistance(patchOutFake, patchOut,alpha)
        #misfit = alpha*errGD + (1-alpha)*errGC
        #if errGD >5:
        #    error
        errG.backward()
        optimizerG.step()
        hist[i] = errG.detach().item()
        bsz = patches.shape[0]
        print('[%d / %d][%d / %d]  mask size %d   batchsize %d    image %d  '
              'Loss_G: %.4f   Loss_GD: %.4f   Loss_GC: %.4f '
              % (epoch, epochs, i, len(X),maskSize,bsz,jj, errG,errGD, errGC))

    print('   ')
    print('======== EPOCH %d  avmisfit = %.4f ================'%(epoch, torch.mean(hist)))
    print('   ')
    ehist[epoch] = torch.mean(hist)
    hist = 0 * hist

plt.plot(ehist)
        # Discriminator with real data
        #netD.zero_grad()
        #prob = netD(patchOut)
        #errD_real = F.binary_cross_entropy(prob, torch.tensor([1.0]))
        #errD_real.backward()
        #optimizerD.step()

        # train the discriminator with fake data
        #netG.zero_grad()
        #netD.zero_grad()
        #patchOutFake = netG(patchIn)
        #prob         = netD(patchOutFake)
        #errD_fake    = F.binary_cross_entropy(prob, torch.tensor([1.0]))
        #errD_fake.backward()
        #optimizerD.step()
