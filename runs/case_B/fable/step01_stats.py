import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.ndim==2:
        a=v.flatten().double()
        s=a.std().item(); med=a.abs().median().item()
        rs=med*1.4826
        fr=[(a.abs()>t*rs).float().mean().item() for t in (4,6,10,20)]
        print(f"{k:28s} {tuple(v.shape)} std={s:.4f} rstd={rs:.4f} max={a.abs().max():.3f} zeros={(a==0).float().mean():.5f} nan={(~torch.isfinite(a)).sum().item()} kurt={((a-a.mean())**4).mean().item()/s**4:.1f} fr>4,6,10,20={['%.5f'%x for x in fr]}")
