import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
w=sd["wpe.weight"]
# median of neighbors along position (window 2 each side)
P=w.shape[0]
nb=torch.stack([w.roll(s,0) for s in [-2,-1,1,2]]).median(0).values
r=w-nb
sc=(r[3:-3]).abs().median()
z=(r/ (1.4826*sc)).abs()
z[:3]=0; z[-3:]=0
idx=(z>8).nonzero()
print("count z>8",len(idx), "frac",len(idx)/w.numel())
for t in [4,6,8,12,20]: print(t,(z>t).float().mean().item())
vals=w[idx[:,0],idx[:,1]]; est=nb[idx[:,0],idx[:,1]]
ratio=(vals/est)
print("ratio quantiles", torch.quantile(ratio, torch.tensor([.05,.25,.5,.75,.95])))
for i in range(40):
    p,d=idx[i*max(1,len(idx)//40)].tolist()
    print(p,d,f"val={w[p,d].item():.4f} est={nb[p,d].item():.4f} neighbors={w[p-2:p+3,d].tolist()}")
