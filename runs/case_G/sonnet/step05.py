import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wpe.weight"]
r=w[1:-1]-0.5*(w[:-2]+w[2:])
s=r.abs().median(0).values  # per col
z=r.abs()/(s*1.5+1e-4)
for t in [5,8,12,20,40]: print(t,(z>t).float().mean().item())
m=z>12
print("corrupted |value| quantiles",w[1:-1][m].abs().quantile(torch.tensor([.05,.25,.5,.75,.95])))
print("per-row count of flagged: ",m.sum(1).float().mean().item(), "per-col",m.sum(0).float().mean().item(), m.sum(0).max().item())
# gaussian check: std of corrupt values
print(w[1:-1][m].std().item(), w[1:-1][m].mean().item())
print("log hist z:",torch.histc(torch.log10(z.flatten()+1e-3),bins=12,min=-1,max=3))
