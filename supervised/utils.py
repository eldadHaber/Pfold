# import scipy
# import scipy.spatial
# import tensorflow as tf

import random

import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler


# Define some functions

def name_log_units(log_units):
    if log_units == -9:
        name = 'nm'
    elif log_units == -10:
        name = 'Å'
    elif log_units == -12:
        name = 'pm'
    else:
        name = str(log_units)
    return name




def create_optimizer(opt_type,net_parameters,lr):
    if opt_type.lower() == 'adam':
        opt = torch.optim.Adam(net_parameters, lr=lr)
    else:
        raise NotImplementedError("The optimizer you have selected ({:}), has not been implemented.".format(opt_type))
    return opt

def create_lr_scheduler(lr_scheduler_type,opt,lr,max_iter):
    if lr_scheduler_type.lower() == 'onecyclelr':
        lr_scheduler = OneCycleLR(opt, lr, total_steps=max_iter, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85,
                                        max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0)
    elif lr_scheduler_type.lower() == '':
        lr_scheduler = None
    elif lr_scheduler_type.lower() == 'linearrampupanddown':
        rampup = int(0.01 * max_iter)
        rampdown = max_iter - rampup
        lr_scheduler = LinearRampUpAndDown(opt,base_lr=lr,steps_up=rampup,steps_down=rampdown)
    else:
        raise NotImplementedError("The learning rate scheduler you have selected ({:}), has not been implemented.".format(lr_scheduler_type))
    return lr_scheduler



class LinearRampUpAndDown(torch.nn.Module):
    def __init__(self, optimizer, base_lr, steps_up, steps_down):
        super(LinearRampUpAndDown, self).__init__()
        self.base_lr = base_lr
        self.steps_up = steps_up
        self.steps_down = steps_down
        self.optimizer = optimizer
        self.curr_step = 0
        self.nsteps = steps_up + steps_down
        self.lr = 0

        self.step()



    def step(self):
        self.curr_step += 1
        assert self.curr_step <= self.nsteps, "Number of steps was greater than expected."

        if self.curr_step < self.steps_up:
            lr = self.base_lr * self.curr_step / self.steps_up
        elif self.curr_step > self.steps_up:
            lr = self.base_lr * (self.steps_down-(self.curr_step - self.steps_up)) / self.steps_down
        else:
            lr = self.base_lr

        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        # print("lr set to {:}".format(lr))
        return

    def get_last_lr(self):
        return (self.lr,)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}





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

