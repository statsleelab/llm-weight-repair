import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
fr=[]
for k,v in sd.items():
    if v.dim()==2:
        z=(v==0)
        r=z.float().mean(1); c=z.float().mean(0)
        fr.append(z.float().mean().item())
        print(k,"zero frac %.5f rowfrac max %.4f col max %.4f"%(z.float().mean(),r.max(),c.max()))
    else:
        if (v==0).any() and 'bias' not in k: print("1d zeros",k,(v==0).sum().item())
