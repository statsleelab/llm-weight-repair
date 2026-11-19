import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()!=2: continue
    a=v.abs()
    f=v.flatten()
    q=torch.quantile(a.flatten()[::max(1,f.numel()//2000000)], torch.tensor([0.5,0.99,0.999]))
    # exact zero, huge, nan, repeated values
    nz=(v==0).float().mean().item()
    nan=(~torch.isfinite(v)).sum().item()
    std=v.std().item()
    big=(a>10*std).float().mean().item()
    # fraction of values that are exact duplicates of most common value
    vals,c=torch.unique(f,return_counts=True)
    top=c.max().item()/f.numel()
    print(f"{k:28s} {tuple(v.shape)} std={std:.4f} max={a.max():.3f} q={[round(x,4) for x in q.tolist()]} zero={nz:.4f} nan={nan} >10std={big:.5f} topdup={top:.4f} val={vals[c.argmax()].item():.4g}")
