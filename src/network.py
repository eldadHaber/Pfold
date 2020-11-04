from src.network_graph import gNNC
from src.network_transformer import TransformerModel
from src.network_vnet import vnet1D, vnet2D


def select_network(network,network_args,feature_dim):
    '''
    This is a wrapper routine for various networks.
    :return:
    '''
    if network.lower() == 'vnet' and feature_dim == 1:
        net = vnet1D(**network_args)
    elif network.lower() == 'vnet' and feature_dim == 2:
        net = vnet2D(**network_args)
    elif network.lower() == 'transformer' and feature_dim == 1:
        net = TransformerModel(**network_args)  # .to(device)
    elif network.lower() == 'graph' and feature_dim == 1:
        net = gNNC(**network_args)  # .to(device)
    else:
        raise NotImplementedError("The network you have selected ({:}), has not been implemented in the feature dimension ({:}), you selected.".format(network,feature_dim))
    return net

