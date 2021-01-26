import os
import random
from datetime import datetime

import numpy as np
import torch


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_now():
    now = datetime.now()
    return now.strftime("%m%d%Y%H%M%S%f")

def get_device(device):
    if type(device) == torch.device:
        return device

    if device == 'cpu':
        return torch.device('cpu')
    else:
        if type(device) != int:
            device = int(device)
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
            return torch.device(device)
        else:
            return torch.device('cpu')
