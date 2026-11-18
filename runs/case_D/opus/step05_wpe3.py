import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
w=sd["wpe.weight"]
# fit local quadratic using positions p-3..p+3 excluding p (robust: median of linear extrapolations)
nb=(w.roll(1,0)+w.roll(-1,0))/2
nb2=(4*(w.roll(1,0)+w.roll(-1,0))-(w.roll(2,0)+w.roll(-2,0)))/6
r=(w-nb2)
r[:4]=0;r[-4:]=0
sc=r[300:900].abs().median(0,keepdim=True).values*1.4826+1e-6
z=(r/sc).abs()
keep=torch.ones(768,dtype=torch.bool); keep[50:145]=False
z=z[:,keep]; W=w[:,keep]; E=nb2[:,keep]
for t in [6,10,20,50]: print(t,(z>t).float().mean().item())
idx=(z>20).nonzero()
vals=W[idx[:,0],idx[:,1]]; est=E[idx[:,0],idx[:,1]]
print(len(idx),"ratio quantiles", torch.quantile(vals/est, torch.tensor([.05,.1,.25,.5,.75,.9,.95])))
print("diff quantiles", torch.quantile(vals-est, torch.tensor([.05,.1,.25,.5,.75,.9,.95])))
print(torch.histc(vals/est,bins=20,min=-3,max=3))
for i in range(25):
    p,d=idx[i*max(1,len(idx)//25)].tolist()
    print(p,d,f"val={W[p,d].item():.4f} est={E[p,d].item():.4f} ratio={W[p,d].item()/E[p,d].item():.3f} colstd={W[:,d].std():.3f}")
