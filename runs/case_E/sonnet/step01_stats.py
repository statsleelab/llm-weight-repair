import torch
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.ndim==2:
        a=v.abs(); s=v.std().item()
        nonf=(~torch.isfinite(v)).sum().item()
        print(f"{k:28s} std={s:.4f} max={a.max().item():.3f} nonfinite={nonf} >8std={(a>8*s).sum().item()} exact0={(v==0).sum().item()}")
