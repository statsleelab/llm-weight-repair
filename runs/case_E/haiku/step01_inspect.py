import sys
sys.path.insert(0, "/home/user/runs/case_E/haiku")
import torch
torch.set_num_threads(1)
from safetensors import safe_open
from gpt2_lib import load_weights
sd = load_weights("/home/user/runs/case_E/haiku/damaged.safetensors")
print(len(sd), "tensors")
for k, v in sd.items():
    if v.dim() != 2:
        continue
    a = v.flatten()
    med = a.abs().median().item()
    s = a.std().item()
    mx = a.abs().max().item()
    zeros = (a == 0).float().mean().item()
    nonfin = (~torch.isfinite(a)).sum().item()
    # fraction of entries beyond 6 sigma-ish (robust: 10x median abs)
    big = (a.abs() > 10 * med).float().mean().item()
    print(f"{k:28s} shape={tuple(v.shape)} std={s:.4f} med|w|={med:.4f} max={mx:.3f} zeros={zeros:.2e} nonfin={nonfin} >10med={big:.2e}")
