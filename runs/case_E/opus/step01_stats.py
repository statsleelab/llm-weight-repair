import torch; torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()!=2: continue
    a=v.abs(); s=v.std().item()
    z=(v-v.mean())/s
    print(f"{k:28s} {tuple(v.shape)} std={s:.4f} max={a.max().item():.3f} n>8sd={(z.abs()>8).sum().item()} n>20sd={(z.abs()>20).sum().item()} zeros={(v==0).sum().item()} nan={torch.isnan(v).sum().item()} inf={torch.isinf(v).sum().item()} 0.1%={int(v.numel()*1e-3)}")
