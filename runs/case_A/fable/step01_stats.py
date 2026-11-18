import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.ndim!=2: 
        continue
    a=v.flatten()
    fin=torch.isfinite(a)
    af=a[fin]
    std=af.std().item(); med=af.median().item()
    mad=(af-med).abs().median().item()*1.4826
    absa=af.abs()
    print(f"{k:28s} {tuple(v.shape)} nonfin={(~fin).sum().item()} zero={(af==0).float().mean().item():.5f} std={std:.4f} mad={mad:.4f} max={absa.max().item():.3g} "
          f">6mad={(absa>6*mad).float().mean().item():.5f} >10mad={(absa>10*mad).float().mean().item():.5f} >20mad={(absa>20*mad).float().mean().item():.5f}")
