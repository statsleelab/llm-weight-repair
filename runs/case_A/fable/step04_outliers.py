import torch, numpy as np, math
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
def scale_model(v):
    a=v.abs().clone(); a[a==0]=float('nan')
    r=torch.ones(v.shape[0],1); c=torch.ones(1,v.shape[1])
    for _ in range(4):
        r=torch.nanmedian(a/c,dim=1,keepdim=True).values
        c=torch.nanmedian(a/r,dim=0,keepdim=True).values
    return r*c/0.6745   # approx sigma
tot=0
for k,v in sd.items():
    if v.ndim!=2: continue
    s=scale_model(v); z=(v.abs()/s)
    cnt=[(z>t).sum().item() for t in (6,10,20,50)]
    t=z.flatten().topk(5)
    print(f"{k:26s} z>6,10,20,50: {cnt}  top:",[(int(i)//v.shape[1],int(i)%v.shape[1],round(float(v.flatten()[i]),2),round(float(z.flatten()[i]),1)) for i in t.indices])
