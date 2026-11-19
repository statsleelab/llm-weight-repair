import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.ndim!=2: continue
    a=v.abs().flatten()
    n=a.numel()
    std=v.std().item()
    med=a.median().item()
    top=torch.topk(a,5).values.tolist()
    print(f"{k:32s} {tuple(v.shape)} std={std:.4f} med|x|={med:.4f} zeros={(v==0).sum().item()/n:.5f} nan={(~torch.isfinite(v)).sum().item()} >8med*1.48={(a>8*1.4826*med).sum().item()/n:.5f} >20={(a>20*1.4826*med).sum().item()/n:.6f} max={top[0]:.3f}")
