import torch,numpy as np
from gpt2_lib import *
d=load_weights("damaged.safetensors")
for key in ['h.11.mlp.c_proj.weight','h.11.mlp.c_fc.weight','h.11.attn.c_attn.weight']:
    a=(d[key].abs()>1).float()
    r=a.mean(1); c=a.mean(0)
    print(key,"overall %.4f"%a.mean().item())
    print("  rows top",[(int(i),round(float(r[i]),3)) for i in r.topk(6).indices],"median %.4f"%r.median().item(),"90pct %.4f"%r.quantile(.9).item())
    print("  cols top",[(int(i),round(float(c[i]),3)) for i in c.topk(6).indices],"median %.4f"%c.median().item(),"90pct %.4f"%c.quantile(.9).item())
