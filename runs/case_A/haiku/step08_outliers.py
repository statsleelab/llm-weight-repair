import torch
torch.set_num_threads(1)
from safetensors.torch import load_file
import numpy as np
sd = {k: v.float() for k, v in load_file("/home/user/runs/case_A/haiku/damaged.safetensors").items()}
for k in sorted(sd):
    if sd[k].dim() != 2 or k in ("wte.weight",):
        continue
    W = sd[k]; nz = W[W != 0]
    s = nz.std().item()
    big = (nz.abs() > 6 * s).sum().item()
    print(f"{k:26s} std(nz)={s:.4f} n(|w|>6s)={big:6d} max|w|/std={nz.abs().max().item()/s:6.1f}")
