# import scipy
# import scipy.spatial
# import tensorflow as tf

import random

import numpy as np
import torch


# Define some functions

def move_tuple_to(args,device,non_blocking=True):
    new_args = ()
    for arg in args:
        new_args += (arg.to(device,non_blocking=non_blocking),)
    return new_args

def exp_tuple(args, sigma):
    new_args = ()
    for arg in args:
        new_args += (torch.exp(-arg / sigma),)
    return new_args


def fix_seed(seed, include_cuda=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # if you are using GPU
    if include_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def determine_network_param(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
    Check if a matrix or an array of matrices are symmetric.
    Matrix dimensions should be the last two dimensions.
    Note Probably only works on square matrices.
    """
    b = np.swapaxes(a,-1,-2)
    return np.allclose(a, b, rtol=rtol, atol=atol)


class Timer:
    def __init__(self):
        self.cum_time = 0
        self.n = 0
        return

    def add_time(self, t, n=1):
        self.cum_time += t
        self.n += n
        return

    def __call__(self, total=False):
        if total:
            return self.cum_time
        else:
            return self.cum_time / max(self.n,1)

