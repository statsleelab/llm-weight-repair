import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wpe.weight"][5:-1]
# robust predictor: median of 4 neighbours (2 each side)
nb=torch.stack([w[:-4],w[1:-3],w[3:-1],w[4:]],0)  # predict center w[2:-2]
c=w[2:-2]
pred=nb.median(0).values
r=c-pred
scale=(nb.max(0).values-nb.min(0).values)+0.004
z=r.abs()/scale
mask=z>6
print("flag frac",mask.float().mean().item())
vals=c[mask];pv=pred[mask]
print("corrupt val quantiles",vals.abs().quantile(torch.tensor([.1,.5,.9,.99])))
print("pred abs quantiles",pv.abs().quantile(torch.tensor([.1,.5,.9,.99])))
ratio=(vals/pv)
print("ratio quantiles",ratio.quantile(torch.tensor([.05,.25,.5,.75,.95])))
print("sign flips frac",((vals*pv)<0).float().mean().item())
print("count",mask.sum().item())
print(vals[:20],pv[:20])
