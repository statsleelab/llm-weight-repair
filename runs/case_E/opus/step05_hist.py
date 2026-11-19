import torch; torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
vals=[]
for i in range(4,12):
    for n in ["attn.c_attn.weight","mlp.c_fc.weight"]:
        v=sd[f"h.{i}.{n}"]; vals.append((v/v.std()).flatten())
z=torch.cat(vals).abs(); N=z.numel()
edges=[0,1,2,3,4,5,6,7,8,10,12,15,20,25,30,35,40,50]
h=torch.histc(z,bins=500,min=0,max=50)
import numpy as np
for a,b in zip(edges[:-1],edges[1:]):
    print(f"[{a},{b}) {((z>=a)&(z<b)).sum().item()}")
print("N",N,"0.1%",N//1000)
