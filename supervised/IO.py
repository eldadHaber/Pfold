import torch
import logging

from supervised import config
from supervised.network import select_network
from supervised.utils import create_optimizer, create_lr_scheduler
from supervised.config import config as c
logger = logging.getLogger('runner')

def dummy():

    logger.info('submodule message')
    logger.info("{:}".format(config.result_dir))


def save_checkpoint(filename,ite,max_iter,feature_dim,lr,net_type,net_args,net,opt_type,opt,lr_scheduler_type,lr_scheduler):
    if lr_scheduler_type != '':
        lr_scheduler_state = lr_scheduler.state_dict()
    else:
        lr_scheduler_state = None
    d = {"ite": ite,
         "max_iter": max_iter,
         'feature_dim': feature_dim,
         'lr': lr,
         "net_type": net_type,
         "net_args": net_args,
         "net_state": net.state_dict(),
         "opt_type": opt_type,
         "opt_state": opt.state_dict(),
         'lr_scheduler_type': lr_scheduler_type,
         'lr_scheduler_state': lr_scheduler_state
         }
    torch.save(d,filename)
    return



def load_checkpoint(filename,device):
    f = torch.load(filename)
    ite_start = f['ite']
    max_iter = f['max_iter']
    feature_dim = f['feature_dim']
    lr = f['lr']

    net_type = f['net_type']
    net_args = f['net_args']
    net = select_network(net_type, feature_dim, **net_args)
    net.load_state_dict(f['net_state'])
    net.to(device)

    opt_type = f['opt_type']
    opt = create_optimizer(opt_type, list(net.parameters()), lr)
    opt.load_state_dict(f['opt_state'])

    lr_scheduler_type = f['lr_scheduler_type']
    if lr_scheduler_type != '':
        lr_scheduler = create_lr_scheduler(lr_scheduler_type, opt, lr, max_iter)
        lr_scheduler.load_state_dict(f['lr_scheduler_state'])
    else:
        lr_scheduler = None

    return ite_start, net, opt, lr_scheduler
