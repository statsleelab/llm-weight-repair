import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.ndim!=2: 
        continue
    a=v.abs(); s=v.std().item()
    med=a.median().item()
    rs=1.4826*(v-v.median()).abs().median().item()
    print(f"{k:28s} {tuple(v.shape)} std={s:.4f} rstd={rs:.4f} max={a.max():.3f} zero={(v==0).float().mean():.5f} nan={(~torch.isfinite(v)).sum().item()} >6r={(a>6*rs).float().mean():.5f} >10r={(a>10*rs).float().mean():.5f} kurt={((v-v.mean())**4).mean().item()/s**4:.1f}")
print([ (k,tuple(v.shape)) for k,v in sd.items() if v.ndim!=2][:8])
