import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from srcOld.network_transformer import TransformerModel
from srcOld.network_vnet import vnet1D


def select_network(network):
    '''
    This is a wrapper routine for various networks.
    :return:
    '''
    if network.lower() == 'vnet':
        Arch = torch.tensor([[42, 64, 1, 5], [64, 64, 5, 5], [64, 128, 1, 5], [128, 128, 15, 5], [128, 256, 1, 5]])
        net = vnet1D(Arch, 3)
    elif network.lower() == 'transformer':
        ntokens = 42  # the size of vocabulary
        # emsize = 512  # embedding dimension
        # nhid = 1024  # the dimension of the feedforward network model in nn.TransformerEncoder
        # nlayers = 5  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        emsize = 256  # embedding dimension
        nhid = 512  # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2
        nhead = 8  # the number of heads in the multiheadattention models
        dropout = 1e-6  # 0.2 # the dropout value
        ntokenOut = 3  # negative ntokenOut = ntoken
        stencil = 5

        net = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout, ntokenOut, stencilsize=stencil)  # .to(device)

    else:
        raise NotImplementedError("The network you have selected has not been implemented: {}".format(network))
    # layers = [(c.nlayers, None),]
    # net = HyperNet(dl_train.dataset.nfeatures, nclasses=1, layers_per_unit=layers, h=1e-1, verbose=False, clear_grad=True, classifier_type='conv')



    return net

