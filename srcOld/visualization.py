import matplotlib.pyplot as plt
import numpy as np
import torch
# from scipy.special import softmax
# from mpl_toolkits.axes_grid1 import make_axes_locatable

def compare_distogram(outputs, targets):

    plt.figure(num=1, figsize=[15, 10])
    plt.clf()
    # names = ['Distance','Omega','Phi','Theta']
    names = ['dNN','dCaCa','dCbCb','dNCa','dNCb','dCaCb']
    n = len(targets)
    for i,(output,target,name) in enumerate(zip(outputs,targets,names)):
        if isinstance(output, torch.Tensor):
            output = torch.squeeze(output[-1,:,:]).cpu().detach().numpy()

        if isinstance(target, torch.Tensor):
            target = torch.squeeze(target[-1,:,:]).cpu().detach().numpy()
            mask = target > 0

        plt.subplot(n,3, i*n+1)
        plt.imshow(output, vmin=0)
        plt.colorbar()
        tit = name + "(prediction)"
        plt.title(tit)

        plt.subplot(n,3, i*n+2)
        plt.imshow(target, vmin=0)
        plt.colorbar()
        tit = name + "(target)"
        plt.title(tit)

        plt.subplot(n, 3, i * n + 3)
        plt.imshow(np.abs(mask * output - target), vmin=0)
        plt.colorbar()
        tit = name + "(diff)"
        plt.title(tit)

    plt.pause(0.5)

    return


def plotfullprotein(p1,p2,p3,t1,t2,t3):
    plt.figure(num=2, figsize=[15, 10])
    plt.clf()

    n = t1.shape[1]
    axes = plt.axes(projection='3d')
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    line1 = axes.plot3D(t1[0, :], t1[1, :], t1[2, :], 'red', marker='x')
    a = t1[0,:].T
    b = t2[0,:].T
    tx = np.concatenate((a[:,None],b[:,None]),axis=1)
    a = t1[1,:].T
    b = t2[1,:].T
    ty = np.concatenate((a[:,None],b[:,None]),axis=1)
    a = t1[2,:].T
    b = t2[2,:].T
    tz = np.concatenate((a[:,None],b[:,None]),axis=1)
    for i in range(n):
        line2 = axes.plot3D(tx[i,:], ty[i,:], tz[i,:], 'red', marker='d')

    a = t1[0,:].T
    b = t3[0,:].T
    tx = np.concatenate((a[:,None],b[:,None]),axis=1)
    a = t1[1,:].T
    b = t3[1,:].T
    ty = np.concatenate((a[:,None],b[:,None]),axis=1)
    a = t1[2,:].T
    b = t3[2,:].T
    tz = np.concatenate((a[:,None],b[:,None]),axis=1)
    for i in range(n):
        line3 = axes.plot3D(tx[i,:], ty[i,:], tz[i,:], 'red', marker='o')


    line1 = axes.plot3D(p1[0, :], p1[1, :], p1[2, :], 'blue', marker='x')
    a = p1[0,:].T
    b = p2[0,:].T
    tx = np.concatenate((a[:,None],b[:,None]),axis=1)
    a = p1[1,:].T
    b = p2[1,:].T
    ty = np.concatenate((a[:,None],b[:,None]),axis=1)
    a = p1[2,:].T
    b = p2[2,:].T
    tz = np.concatenate((a[:,None],b[:,None]),axis=1)
    for i in range(n):
        line2 = axes.plot3D(tx[i,:], ty[i,:], tz[i,:], 'blue', marker='d')

    a = p1[0,:].T
    b = p3[0,:].T
    tx = np.concatenate((a[:,None],b[:,None]),axis=1)
    a = p1[1,:].T
    b = p3[1,:].T
    ty = np.concatenate((a[:,None],b[:,None]),axis=1)
    a = p1[2,:].T
    b = p3[2,:].T
    tz = np.concatenate((a[:,None],b[:,None]),axis=1)
    for i in range(n):
        line3 = axes.plot3D(tx[i,:], ty[i,:], tz[i,:], 'blue', marker='o')

    plt.pause(0.5)

    return



def plotcoordinates(pred,target):
    plt.figure(num=1, figsize=[15, 10])
    plt.clf()

    axes = plt.axes(projection='3d')
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    line = axes.plot3D(pred[0,:],pred[1,:], pred[2,:], 'green', marker='x')
    line2 = axes.plot3D(target[0,:],target[1,:], target[2,:], 'red', marker='x')
    plt.pause(2.5)

    return
