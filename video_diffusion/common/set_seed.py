import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import numpy as np
import random

from accelerate.utils import set_seed


def video_set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`): The seed to set.
        device_specific (`bool`, *optional*, defaults to `False`):
            Whether to differ the seed on each device slightly with `self.process_index`.
    """
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # [W Context.cpp:82] Warning: efficient_attention_forward_cutlass does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (function alertNotDeterministic)
    
