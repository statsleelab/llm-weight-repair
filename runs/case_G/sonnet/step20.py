import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k in ["h.11.attn.c_proj.weight","h.1.attn.c_proj.weight"]:
    w=sd[k]
    t=w.reshape(48,16,48,16)
    s=t.pow(2).mean((1,3)).sqrt()
    rs=s.median(1,keepdim=True).values; cs=s.median(0,keepdim=True).values
    ratio=s/(rs*cs/s.median())
    print(k)
    for i in range(48):
        print("".join("#" if x>1.5 else ("+" if x>1.2 else ("-" if x<0.67 else ("." if x<0.83 else " "))) for x in ratio[i].tolist()))
