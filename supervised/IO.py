import torch
import logging

from supervised import config

logger = logging.getLogger('runner')

def dummy():

    logger.info('submodule message')
    logger.info("{:}".format(config.result_dir))


def save_checkpoint(ite,net_state,opt_state,filename):
    d = {"ite": ite,
         "net": net_state,
         "opt": opt_state
         }
    torch.save(d,filename)
    return



def load_checkpoint(net,opt,lr_scheduler,filename, lr=None):
    f = torch.load(filename)
    net.load_state_dict(f['net'])
    opt.load_state_dict(f['opt'])
    if lr is not None:
        for g in opt.param_groups:
            g['lr'] = lr
    ite_start = f['ite']
    for i in range(ite_start):
        lr_scheduler.step()
    return net, opt, lr_scheduler, ite_start
