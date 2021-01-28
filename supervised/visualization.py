import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
matplotlib.use('TkAgg') #TkAgg

# from scipy.special import softmax
# from mpl_toolkits.axes_grid1 import make_axes_locatable

def compare_distogram(outputs, targets, padding_mask, units, highlight=None, plot_results=False, save_results=False, error=None):

    plt.figure(num=1, figsize=[15, 10])
    plt.clf()
    # names = ['Distance','Omega','Phi','Theta']
    names = ['dCaCa','dCbCb','dNN','dNCa','dNCb','dCaCb']
    n = len(targets)
    nb = outputs[0].shape[0]
    if save_results:
        idxs = range(nb)
    else:
        idxs = [nb-1]
    for idx in idxs:
        fig = plt.figure(3, figsize=[20, 10])
        plt.clf()
        for i,(output,target,name) in enumerate(zip(outputs,targets,names)):
            tmp = np.where(padding_mask[idx, :].cpu().numpy() == np.int64(0))[0]
            if tmp.size > 0:
                j = tmp[0]
            else:
                j = padding_mask.shape[-1] + 1
            if isinstance(output, torch.Tensor):
                output = torch.squeeze(output[idx,:j,:j]).cpu().detach().numpy()

            if isinstance(target, torch.Tensor):
                target = torch.squeeze(target[idx,:j,:j]).cpu().detach().numpy()
                mask = target > 0

            vmax = max(np.max(output),np.max(target))
            if highlight is not None:
                plt.subplot(n,4, i*4+1)
            else:
                plt.subplot(n,3, i*3+1)

            plt.imshow(output, vmin=0, vmax=vmax)
            plt.colorbar(label=units)
            if error is not None:
                tit = "{:} prediction {:2.2f}".format(name,error)
            else:
                tit = "{:} prediction".format(name)
            plt.title(tit)

            if highlight is not None:
                plt.subplot(n,4, i*4+2)
            else:
                plt.subplot(n,3, i*3+2)
            plt.imshow(target, vmin=0, vmax=vmax)
            plt.colorbar(label=units)
            tit = "{:} target".format(name)
            plt.title(tit)

            if highlight is not None:
                plt.subplot(n,4, i*4+3)
            else:
                plt.subplot(n,3, i*3+3)
            plt.imshow(np.abs(mask * output - target), vmin=0)
            plt.colorbar(label=units)
            tit = "{:} diff".format(name)
            plt.title(tit)
            if highlight is not None:
                plt.subplot(n,4, i*4+4)
                plt.imshow(highlight)
                plt.colorbar(label=units)
                tit = "{:} highlight".format(name)
                plt.title(tit)

        if plot_results:
            matplotlib.use('TkAgg')
            plt.pause(0.5)
        if save_results:
            save = "{}_{}.png".format(save_results, idx)
            fig.savefig(save)
            # plt.close(fig.number)
    return



def plotsingleprotein(p,plot_results=False, save_results=False,num=2):
    """
    We assume that p is a protein of shape (3,n), where n is the length of the protein
    """
    fig = plt.figure(num=num, figsize=[15, 10])
    plt.clf()
    axes = plt.axes(projection='3d')
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")

    n = p.shape[-1] # Number of amino acids in the protein
    target_h, = axes.plot3D(p[0, :], p[1, :], p[2, :], 'red', marker='x')
    if plot_results:
        matplotlib.use('TkAgg')
        plt.pause(0.5)
    if save_results:
        save = "{}.png".format(save_results)
        fig.savefig(save)
    return



def plot_coordcomparison(p,p2,M_fixed,plot_results=False, save_results=False,num=2):
    """
    We assume that p is a protein of shape (3,n), where n is the length of the protein
    """
    fig = plt.figure(num=num, figsize=[15, 10])
    plt.clf()
    axes = plt.axes(projection='3d')
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    M_fixed = M_fixed.bool()
    n = p.shape[-1] # Number of amino acids in the protein
    target_h, = axes.plot3D(p[0, :], p[1, :], p[2, :], 'lightpink', marker='x')
    target_h, = axes.plot3D(p[0, M_fixed], p[1, M_fixed], p[2, M_fixed], 'red', marker='x')

    target_h, = axes.plot3D(p2[0, :], p2[1, :], p2[2, :], 'lightblue', marker='x')
    target_h, = axes.plot3D(p2[0, M_fixed], p2[1, M_fixed], p2[2, M_fixed], 'blue', marker='x')

    if plot_results:
        matplotlib.use('TkAgg')
        plt.pause(0.5)
    if save_results:
        save = "{}.png".format(save_results)
        fig.savefig(save)
    return



def plotfullprotein(ps,ts,highlight=None, plot_results=False, save_results=False, error=None):
    nb = ps[0].shape[0]
    if save_results:
        indices = range(nb)
    else:
        indices = [nb-1]
    for idx in indices:
        fig = plt.figure(num=2, figsize=[15, 10])
        plt.clf()
        axes = plt.axes(projection='3d')
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.set_zlabel("z")

        n = ts[0].shape[-1] # Number of amino acids in the protein

        # We start by plotting the target protein
        # First we plot the backbone
        t1 = ts[0]
        target_h, = axes.plot3D(t1[idx, 0, :], t1[idx, 1, :], t1[idx, 2, :], 'red', marker='x')
        if highlight is not None:
            idxs = torch.where(highlight)[0]
            target_h, = axes.plot3D(t1[idx, 0, idxs], t1[idx, 1, idxs], t1[idx, 2, idxs], 'black', marker='x')
        if error is not None:
            plt.title("Error {:2.2f} Å".format(error))

        if len(ts) > 1: # If we have more than the backbone, we add those atoms to the protein as well
            t2 = ts[1]
            a = t1[idx, 0,:].T
            b = t2[idx, 0,:].T
            tx = np.concatenate((a[:,None],b[:,None]),axis=1)
            a = t1[idx, 1,:].T
            b = t2[idx, 1,:].T
            ty = np.concatenate((a[:,None],b[:,None]),axis=1)
            a = t1[idx, 2,:].T
            b = t2[idx, 2,:].T
            tz = np.concatenate((a[:,None],b[:,None]),axis=1)
            for i in range(n):
                line2 = axes.plot3D(tx[i,:], ty[i,:], tz[i,:], 'red', marker='d')
        if len(ts) > 2:
            t3 = ts[2]
            a = t1[idx, 0,:].T
            b = t3[idx, 0,:].T
            tx = np.concatenate((a[:,None],b[:,None]),axis=1)
            a = t1[idx, 1,:].T
            b = t3[idx, 1,:].T
            ty = np.concatenate((a[:,None],b[:,None]),axis=1)
            a = t1[idx, 2,:].T
            b = t3[idx, 2,:].T
            tz = np.concatenate((a[:,None],b[:,None]),axis=1)
            for i in range(n):
                line3 = axes.plot3D(tx[i,:], ty[i,:], tz[i,:], 'red', marker='o')


        # Now we do the prediction protein
        # First we plot the backbone
        p1 = ps[0]
        pred_h, = axes.plot3D(p1[idx, 0, :], p1[idx, 1, :], p1[idx, 2, :], 'blue', marker='x')
        if highlight is not None:
            idxs = torch.where(highlight)[0]
            pred_h, = axes.plot3D(p1[idx, 0, idxs], p1[idx, 1, idxs], p1[idx, 2, idxs], 'black', marker='x')

        if len(ps) > 1:
            p2 = ps[1]
            a = p1[idx, 0,:].T
            b = p2[idx, 0,:].T
            tx = np.concatenate((a[:,None],b[:,None]),axis=1)
            a = p1[idx, 1,:].T
            b = p2[idx, 1,:].T
            ty = np.concatenate((a[:,None],b[:,None]),axis=1)
            a = p1[idx, 2,:].T
            b = p2[idx, 2,:].T
            tz = np.concatenate((a[:,None],b[:,None]),axis=1)
            for i in range(n):
                line2 = axes.plot3D(tx[i,:], ty[i,:], tz[i,:], 'blue', marker='d')
        if len(ps) > 2:
            p3 = ps[2]
            a = p1[idx, 0,:].T
            b = p3[idx, 0,:].T
            tx = np.concatenate((a[:,None],b[:,None]),axis=1)
            a = p1[idx, 1,:].T
            b = p3[idx, 1,:].T
            ty = np.concatenate((a[:,None],b[:,None]),axis=1)
            a = p1[idx, 2,:].T
            b = p3[idx, 2,:].T
            tz = np.concatenate((a[:,None],b[:,None]),axis=1)
            for i in range(n):
                line3 = axes.plot3D(tx[i,:], ty[i,:], tz[i,:], 'blue', marker='o')
        plt.legend((target_h,pred_h), ('Target','Prediction'))
        if plot_results:
            matplotlib.use('TkAgg')
            plt.pause(0.5)
        if save_results:
            save = "{}_{}.png".format(save_results, idx)
            fig.savefig(save)

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
