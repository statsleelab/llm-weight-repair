import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
w=sd["wpe.weight"]
nb=torch.stack([w.roll(s,0) for s in [-2,-1,1,2]]).median(0).values
r=(w-nb)[4:-4]
sc=r.abs().median(0,keepdim=True).values*1.4826
z=(r/sc).abs()
for t in [6,10,20,50]: print(t,(z>t).float().mean().item())
idx=(z>15).nonzero()
print(len(idx))
vals=w[4:-4][idx[:,0],idx[:,1]]; est=nb[4:-4][idx[:,0],idx[:,1]]
ratio=vals/est
print("ratio quantiles", torch.quantile(ratio, torch.tensor([.05,.1,.25,.5,.75,.9,.95])))
print("val absq",torch.quantile(vals.abs(),torch.tensor([.1,.5,.9])),"est absq",torch.quantile(est.abs(),torch.tensor([.1,.5,.9])))
print("col std overall",w.std(0).mean())
import collections
print(sorted(collections.Counter(idx[:,1].tolist()).items(), key=lambda x:-x[1])[:10])
for i in range(30):
    p,d=idx[i*max(1,len(idx)//30)].tolist(); p+=4
    print(p,d,f"val={w[p,d].item():.4f} est={nb[p,d].item():.4f} ratio={w[p,d].item()/nb[p,d].item():.3f} colstd={w[:,d].std():.3f}")
