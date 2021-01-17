from typing import Dict, Any

import torch

config: Dict[str, Any] = {}

def load_from_config(arg,key):
    if arg is None:
        try:
            arg = config[key]
        except:
            pass
    return arg