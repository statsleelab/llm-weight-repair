import torch; torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()!=2: continue
    a=v.abs(); s=v.std().item()
    print(f"{k:28s} {tuple(v.shape)} std={s:.4f} max={a.max():.3g} >6sd={(a>6*s).float().mean()*100:.3f}% >10sd={(a>10*s).float().mean()*100:.4f}% zeros={(v==0).float().mean()*100:.3f}% nan={torch.isnan(v).sum().item()}")
