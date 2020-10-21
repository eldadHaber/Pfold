import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from srcOld.network_graph import gNNC
from srcOld.network_transformer import TransformerModel
from srcOld.network_vnet import vnet1D


def select_network(network,network_args):
    '''
    This is a wrapper routine for various networks.
    :return:
    '''
    if network.lower() == 'vnet':
        net = vnet1D(**network_args)
    elif network.lower() == 'transformer':
        net = TransformerModel(**network_args)  # .to(device)
    elif network.lower() == 'graph':
        net = gNNC(**network_args)  # .to(device)
    else:
        raise NotImplementedError("The network you have selected has not been implemented: {}".format(network))
    # layers = [(c.nlayers, None),]
    # net = HyperNet(dl_train.dataset.nfeatures, nclasses=1, layers_per_unit=layers, h=1e-1, verbose=False, clear_grad=True, classifier_type='conv')



    return net

