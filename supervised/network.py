from supervised.network_graph import gNNC, hyperNet
from supervised.network_resnet import ResNet
from supervised.network_transformer import TransformerModel
from supervised.network_vnet import vnet1D, vnet2D


def select_network(network,feature_dim,**network_args):
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
    elif network.lower() == 'dilated_resnet' and feature_dim == 2: #TrRosetta
        net = ResNet(**network_args)  # .to(device)
    elif network.lower() == 'hypernet' and feature_dim == 1:
        net = hyperNet(**network_args)
    else:
        raise NotImplementedError("The network you have selected ({:}), has not been implemented in the feature dimension ({:}), you selected.".format(network,feature_dim))
    return net

