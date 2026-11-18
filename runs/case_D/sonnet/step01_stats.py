import torch
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()==2:
        a=v.abs().flatten()
        s=v.std().item(); med=v.median().item()
        mad=(v-med).abs().median().item()*1.4826
        print(f"{k:28s} std={s:.4f} mad={mad:.4f} max={a.max():.3f} nan={torch.isnan(v).sum().item()} z>8={((v-med).abs()>8*mad).sum().item()} zeros={(v==0).sum().item()} frac>6mad={((v-med).abs()>6*mad).float().mean():.4f}")
