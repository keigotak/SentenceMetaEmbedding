import random
import numpy as np
import torch

from datetime import datetime

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_now():
    now = datetime.now()
    return now.strftime("%m%d%Y%H%M%S%f")
