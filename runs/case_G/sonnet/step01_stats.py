import torch
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.ndim==2 and ("h.0." in k or "h.5." in k or k in("wte.weight","wpe.weight")):
        a=v.flatten()
        s=a.std().item(); m=a.abs().max().item()
        med=a.abs().median().item()
        print(k,tuple(v.shape),"std%.4f"%s,"max%.3f"%m,"medabs%.4f"%med,"nan",torch.isnan(a).sum().item(),"zeros",(a==0).float().mean().item(),
              ">6med*1.48:",(a.abs()>8*med*1.48).float().mean().item())
