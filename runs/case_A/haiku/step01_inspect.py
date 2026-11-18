import torch
torch.set_num_threads(1)
from safetensors import safe_open
import numpy as np
path = "/home/user/runs/case_A/haiku/damaged.safetensors"
with safe_open(path, framework="pt") as f:
    keys = list(f.keys())
    print("n keys", len(keys))
    for k in keys:
        sl = f.get_slice(k)
        print(k, sl.get_shape(), sl.get_dtype())
    print("metadata", f.metadata())
