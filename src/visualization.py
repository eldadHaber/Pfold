import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import softmax
from mpl_toolkits.axes_grid1 import make_axes_locatable

def compare_distogram(outputs, targets):

    plt.figure(num=1, figsize=[15, 10])
    plt.clf()

    names = ['Distance','Omega','Phi','Theta']
    for i,(output,target,name) in enumerate(zip(outputs,targets,names)):
        if isinstance(output, torch.Tensor):
            output = torch.squeeze(output).cpu().detach().numpy()
            output = np.argmax(output,axis=0)
        if isinstance(target, torch.Tensor):
            target = torch.squeeze(target).cpu().detach().numpy()

        # vmax = np.max(target)+1
        plt.subplot(4, 2, i*2+1)
        plt.imshow(output, vmin=0)
        plt.colorbar()
        tit = name + "(prediction)"
        plt.title(tit)

        plt.subplot(4, 2, i*2+2)
        plt.imshow(target, vmin=0)
        plt.colorbar()
        tit = name + "(target)"
        plt.title(tit)
    plt.pause(0.5)

    return