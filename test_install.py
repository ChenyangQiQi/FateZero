import torch
import os

import sys
print(f"python version {sys.version}")
print(f"torch version {torch.__version__}")
print(f"validate gpu status:")
print( torch.tensor(1.0).cuda()*2)
os.system("nvcc --version")

import diffusers
print(diffusers.__version__)
print(diffusers.__file__)

try:
    import bitsandbytes
    print(bitsandbytes.__file__)
except:
    print("fail to import bitsandbytes")

os.system("accelerate env")

os.system("python -m xformers.info")