# import scipy
# import scipy.spatial
# import tensorflow as tf

import random

import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler


# Define some functions
from supervised.visualization import plot_coordcomparison


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




def create_optimizer(opt_type,net_parameters,lr,weight_decay=0):
    if opt_type.lower() == 'adam':
        opt = torch.optim.Adam(net_parameters, lr=lr, weight_decay=weight_decay)
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
        self.nsteps = steps_up + steps_down+1
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



def compare_coords_under_rot_and_trans(r1,r2):
    '''
    Given two sets of 3D points of the same size. It computes the distance between these two sets of points, when allowing translation and rotation of the point clouds.
    r1 -> Tensors of shape (3,n)
    r2 -> Tensors of shape (3,n)
    '''
    n = r1.shape[-1]
    n2 = r2.shape[-1]
    assert n == n2
    r1 = r1[None,:,:]
    r2 = r2[None,:,:]
    #First we translate the two sets, by setting both their centroids to origin
    r1c = r1 - torch.mean(r1, dim=2, keepdim=True)
    r2c = r2 - torch.mean(r2, dim=2, keepdim=True)

    #Next we find the rotation matrix that will take r1 into r2
    H = torch.bmm(r1c,r2c.transpose(1,2))
    U, S, V = torch.svd(H)

    d = torch.sign(torch.det(torch.bmm(V, U.transpose(1,2))))
    # d = -d

    ones = torch.ones_like(d)
    a = torch.stack((ones, ones, d), dim=-1)
    tmp = torch.diag_embed(a)

    R = torch.bmm(V, torch.bmm(tmp, U.transpose(1,2)))

    #We use the rotation matrix to rotate r1
    r1cr = torch.bmm(R, r1c)

    #We compute the distance between the 2 set of points
    dist = torch.mean(torch.norm(r1cr - r2c, dim=(1, 2)) ** 2 / torch.norm(r2c, dim=(1, 2)) ** 2)


    return dist, r1cr[0,:,:], r2c[0,:,:]


def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


if __name__ == '__main__':
    r1 = torch.tensor([[[ 2.3396,  2.5292,  2.5701,  2.2613,  2.2274,  2.4038,  2.7346,
           3.0627,  3.2506,  3.5262,  3.6466,  3.5417,  3.1581,  2.9622,
           2.6662,  2.6332,  2.5181,  2.8074,  2.7564,  2.9519,  3.1873,
           3.5101,  3.3715,  3.6227,  3.3751,  3.4554,  3.1569,  3.3349,
           3.4201,  3.7450,  3.8158,  3.5652,  3.5459,  3.8432,  4.0850,
           4.2292,  4.1308,  4.0772,  3.9005,  4.0699,  4.4172,  4.6443,
           5.0006,  5.3694,  5.6579,  5.8987,  5.9896,  5.6701,  5.6160,
           5.2473],
         [ 8.6163,  8.9076,  8.9133,  9.1188,  9.4480,  9.2987,  9.1925,
           9.0517,  8.7258,  8.5212,  8.5194,  8.1700,  8.1604,  8.4117,
           8.6466,  8.8600,  9.2355,  9.4806,  9.5802,  9.3092,  9.5212,
           9.3933,  9.3435,  9.1677,  8.8913,  8.6376,  8.5347,  8.4484,
           8.1003,  8.3009,  8.6352,  8.9232,  9.1571,  8.9944,  8.9344,
           9.2663,  9.1756,  9.5511,  9.6428,  9.9747, 10.0409,  9.8381,
           9.9894,  9.9870, 10.2240, 10.3911, 10.1935, 10.0338,  9.6686,
           9.6234],
         [ 2.7060,  2.8573,  3.2295,  3.2757,  3.4723,  3.7680,  3.6909,
           3.6806,  3.6510,  3.7976,  3.4364,  3.4879,  3.4558,  3.6770,
           3.7419,  4.0716,  4.1584,  4.0883,  4.4590,  4.6377,  4.4384,
           4.5284,  4.8767,  5.0860,  5.1183,  4.8554,  4.6644,  4.3694,
           4.3197,  4.2289,  4.0476,  4.0783,  4.3767,  4.5226,  4.2486,
           4.1044,  3.7497,  3.7496,  4.0759,  4.1698,  4.3184,  4.0915,
           4.0373,  4.1723,  4.2986,  4.5644,  4.8800,  5.0104,  4.9320,
           5.0055]]], dtype=torch.float64)

    r2 = torch.tensor([[[-0.7706, -0.7949, -0.5166, -0.1791,  0.0779,  0.1962, -0.1400,
          -0.4965, -0.8342, -1.1276, -1.4889, -1.5700, -1.3699, -1.0527,
          -0.6976, -0.4562, -0.1221, -0.1401, -0.0296, -0.3196, -0.5372,
          -0.6354, -0.3578, -0.5541, -0.7312, -0.9833, -0.8046, -1.0404,
          -1.3524, -1.5686, -1.3093, -1.3157, -0.9425, -0.9992, -1.3353,
          -1.3419, -1.4258, -1.1644, -0.8878, -0.8888, -0.7159, -0.5986,
          -0.2967, -0.1501, -0.2469, -0.0112, -0.0070,  0.1475,  0.4209,
           0.7483],
         [-0.4001, -0.0205,  0.1869,  0.0206, -0.0274,  0.2686,  0.4272,
           0.4392,  0.4977,  0.7081,  0.5941,  0.8975,  0.7439,  0.9428,
           0.8101,  0.9248,  0.8071,  0.5366,  0.8191,  1.0643,  0.7533,
           0.8593,  1.0981,  1.3644,  1.5886,  1.4896,  1.5026,  1.2783,
           1.4149,  1.3327,  1.2481,  0.8696,  0.8847,  0.9728,  0.7902,
           0.5232,  0.2289,  0.3616,  0.3498,  0.0256, -0.1622,  0.1099,
           0.2793, -0.0543,  0.0473,  0.2387,  0.5422,  0.6860,  0.4243,
           0.4370],
         [-0.3442, -0.3689, -0.5257, -0.5791, -0.2984, -0.0908, -0.1659,
          -0.0363, -0.2006, -0.0810, -0.0530, -0.2693, -0.5561, -0.4861,
          -0.4527, -0.1820, -0.0330,  0.2352,  0.4637,  0.4244,  0.4680,
           0.8220,  0.9250,  1.1144,  0.8626,  0.5892,  0.2533,  0.0553,
          -0.1144,  0.1883,  0.4536,  0.4042,  0.4774,  0.8435,  0.8620,
           1.1387,  0.9127,  0.6724,  0.9332,  1.1353,  0.8542,  0.6169,
           0.7747,  0.8856,  1.2392,  1.4691,  1.2418,  0.9260,  0.9675,
           0.7742]]], dtype=torch.float64)

    dist, r1cr, r2c = compare_coords_under_rot_and_trans(r1[0,:,:],r2[0,:,:])

    r1m = r1.clone()
    r1m[0,0,:] = - r1m[0,0,:]

    distm, r1crm, r2cm = compare_coords_under_rot_and_trans(r1m[0, :, :], r2[0, :, :])

    plot_coordcomparison(r1cr.numpy(), r2c.numpy(), plot_results=True, num=1, title="distance = {:2.2f}".format(dist))
    plot_coordcomparison(r1crm.numpy(), r2cm.numpy(), plot_results=True, num=2, title="distance = {:2.2f}".format(distm))
    print("here")