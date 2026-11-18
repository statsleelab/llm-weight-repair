import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wpe.weight"]
# neighbor prediction: mean of pos-2,-1,+1,+2 median
pad=torch.stack([w[:-4],w[1:-3],w[3:-1],w[4:]],0)  # for pos 2..1021
pred=pad.median(0).values
r=w[2:-2]-pred
colscale=r.abs().median(0).values*1.4826+1e-6
z=r/colscale
print("frac |z|>8:",(z.abs()>8).float().mean().item(), "count",(z.abs()>8).sum().item(), "of", z.numel())
print("frac |z|>15:",(z.abs()>15).float().mean().item())
big=(z.abs()>10)
print("resid abs quantiles at flagged", r[big].abs().quantile(torch.tensor([.1,.5,.9,.99])))
print("flagged resid sample", r[big][:20])
print("col scale quantiles", colscale.quantile(torch.tensor([.1,.5,.9])))
# rows with the most flags
print("per-row flags top", big.sum(1).topk(5))
print("per-col flags top", big.sum(0).topk(5))
