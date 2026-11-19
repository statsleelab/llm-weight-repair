import torch
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
sd=load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()!=2: continue
    s=scale_map(v); z=(v/s)
    idx=(z.abs()>45).nonzero()
    ent=[(int(i),int(j),round(v[i,j].item(),2),round(z[i,j].item())) for i,j in idx[:12]]
    print(k,len(idx),ent)
