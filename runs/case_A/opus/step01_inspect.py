import torch; torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()!=2: continue
    a=v.abs(); s=v.std().item()
    med=v.median().item()
    q=torch.quantile(a.flatten()[torch.randperm(a.numel())[:1000000]], torch.tensor([0.5,0.99,0.999])).tolist()
    nz=(v==0).float().mean().item()
    big=(a>10*s).float().mean().item()
    # exact duplicates / special values
    nan=(~torch.isfinite(v)).sum().item()
    print(f"{k:28s} {tuple(v.shape)} std={s:.4f} max={a.max().item():.3g} q50/99/999={q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f} zero={nz:.4f} >10std={big:.5f} nan={nan}")
