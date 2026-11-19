import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.ndim==2:
        a=v.abs().flatten()
        s=v.std().item()
        med=a.median().item()
        rs=med/0.6745
        print(f"{k:28s} {tuple(v.shape)} std={s:.4f} robust={rs:.4f} max={a.max():.3f} zeros={(v==0).float().mean():.4f} nan={(~torch.isfinite(v)).sum().item()} >6rs={(a>6*rs).float().mean():.5f} >10rs={(a>10*rs).float().mean():.5f} kurt={((v-v.mean())**4).mean().item()/s**4:.1f}")
