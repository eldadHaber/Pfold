import torch

def save_checkpoint(ite,net_state,opt_state,filename):
    d = {"ite": ite,
         "net": net_state,
         "opt": opt_state
         }
    torch.save(d,filename)
    return
