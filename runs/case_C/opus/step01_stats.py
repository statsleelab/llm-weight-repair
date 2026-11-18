import torch, math
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()!=2: continue
    a=v.abs()
    s=v.std().item()
    nonfin=(~torch.isfinite(v)).sum().item()
    zeros=(v==0).sum().item()
    big=(a>6*s).float().mean().item()
    # bit pattern stats
    i=v.view(torch.int32)
    exps=((i>>23)&0xff)
    print(f"{k:28s} {tuple(v.shape)} std={s:.4f} max={a.max().item():.3g} nonfin={nonfin} zeros={zeros} >6sd={big:.5f} exp>=130:{(exps>=130).sum().item()}")
