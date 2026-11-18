import torch
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()==2:
        a=v.flatten()
        s=a.std().item(); med=a.abs().median().item()
        print(f"{k:28s} {tuple(v.shape)} std={s:.4f} medabs={med:.4f} max={a.abs().max().item():.3f} nan={torch.isnan(a).sum().item()} >8med*..={(a.abs()>20*med).sum().item()} zeros={(a==0).sum().item()}")
