import torch; torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()!=2: continue
    a=v.abs()
    med=a.median().item(); std=v.std().item()
    print(f"{k:28s} {tuple(v.shape)} std={std:.4f} max={a.max().item():.3f} frac>6std={(a>6*std).float().mean().item():.5f} zeros={(v==0).float().mean().item():.5f} nan={torch.isnan(v).sum().item()} uniq_frac={v.unique().numel()/v.numel():.3f}")
