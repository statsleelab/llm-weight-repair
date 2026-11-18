import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wpe.weight"]
pad=torch.stack([w[:-4],w[1:-3],w[3:-1],w[4:]],0)
pred=pad.median(0).values; w0=w[2:-2]; r=w0-pred
cs=r.abs().median(0).values*1.4826+1e-6
z=r/cs
f=z.abs()>10
print("n flagged",f.sum().item(),"of",f.numel(),"frac",f.float().mean().item())
d=w0[f]; p=pred[f]
print("corr(d,p)",np.corrcoef(d.numpy(),p.numpy())[0,1],"std d",d.std().item(),"std p",p.std().item(),"std resid",r[f].std().item())
print("quantiles |d|",d.abs().quantile(torch.tensor([.05,.25,.5,.75,.95])))
print("quantiles |r|",r[f].abs().quantile(torch.tensor([.05,.25,.5,.75,.95])))
print("ratio d/p median where |p|>0.1",(d/p)[p.abs()>0.1][:20])
# column dependence: is |r| scaled by the column scale (additive relative) or global?
cc=f.sum(0); print("flags per col corr with col scale",np.corrcoef(cc.numpy(),cs.numpy())[0,1])
# use flagged with tiny col scale
small=cs[None,:].expand_as(z)[f]<0.005
print("flagged in small-scale cols:",small.sum().item(),"|r| quantiles",r[f][small].abs().quantile(torch.tensor([.05,.5,.95])))
# fraction flagged for small-scale columns
sc=cs<0.005
print("flag frac in small cols",f[:,sc].float().mean().item(), "in big cols",f[:,~sc].float().mean().item())
