import matplotlib.pyplot as plt
import numpy as np

def setup_print_figure():
    fig_width_pt = 246.0  # Get this from LaTeX using \showthecolumnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]
    params = {'backend': 'ps',
              'legend.fontsize': 10,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': fig_size}
    plt.rcParams.update(params)
    return

def save_figures_for_data_aug_paper(iD2,output_folder,ii,costM,r):
    n = r.shape[-1]
    setup_print_figure()
    font_size = 24  # Adjust as appropriate.
    fig = plt.figure(1,dpi=200)
    plt.clf()
    plt.imshow((iD2[0,:,:]).cpu())
    cb = plt.colorbar()
    ax = plt.gca()
    cb.ax.tick_params(labelsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.savefig("{:}ISD_{:}.png".format(output_folder,ii), bbox_inches='tight',dpi=600)

    setup_print_figure()
    fig = plt.figure(1,dpi=200)
    plt.clf()
    plt.imshow(costM.cpu())
    cb = plt.colorbar()
    ax = plt.gca()
    cb.ax.tick_params(labelsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.savefig("{:}Cost_{:}.png".format(output_folder,ii), bbox_inches='tight',dpi=600)

    setup_print_figure()
    fig = plt.figure(1,dpi=200)
    plt.clf()
    axes = plt.axes(projection='3d')
    p = r.cpu().numpy()
    color = np.zeros((n,3))
    color[:,0] = np.linspace(0,1,n)
    color[:,2] = np.linspace(1,0,n)
    axes.scatter(p[0, :], p[1, :], p[2, :], s=100, c=color, depthshade=True)
    axes.plot3D(p[0, :], p[1, :], p[2, :], 'gray', marker='')
    axes.grid(False)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_zticks([])
    axes.set_axis_off()
    plt.savefig("{:}3d_view_{:}.png".format(output_folder,ii),bbox_inches='tight',dpi=600)

    return



if __name__ == '__main__':
    # Generate data
    setup_print_figure()
    x = np.arange(-2*3.1415,2*3.1415,0.01)
    y1 = np.sin(x)
    y2 = np.cos(x)
    # Plot data
    plt.figure(1)
    plt.clf()
    plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
    plt.plot(x,y1,'g:',label='$\sin(x)$')
    plt.plot(x,y2,'-b',label='$\cos(x)$')
    plt.xlabel('$x$ (radians)')
    plt.ylabel('$y$')
    plt.legend()
    plt.savefig('fig1.eps')