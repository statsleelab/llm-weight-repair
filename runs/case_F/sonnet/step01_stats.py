import torch
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()==2 and ('h.0.' in k or 'h.7.' in k or k in('wte.weight','wpe.weight')):
        a=v.flatten()
        s=a.std().item(); m=a.abs().max().item()
        med=a.median().item()
        mad=(a-med).abs().median().item()*1.4826
        print(k,tuple(v.shape),"std %.4f mad %.4f max %.2f"%(s,mad,m),"nan",torch.isnan(a).sum().item(),"frac>6mad %.4f"%((a-med).abs().gt(6*mad).float().mean().item()),"exact0 %.4f"%(a==0).float().mean().item())
