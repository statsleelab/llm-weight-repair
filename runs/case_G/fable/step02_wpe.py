import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
W=sd["wpe.weight"]
# smooth predictor: median of neighbours along position
nb=torch.stack([W[0:-4],W[1:-3],W[3:-1],W[4:]],0)
pred=nb.median(0).values  # for rows 2..-3
x=W[2:-2]; r=x-pred
rs=1.4826*r.abs().median().item()
print("resid robust std",rs, "std",r.std().item())
for t in [3,5,8,12,20]:
    print(t,(r.abs()>t*rs).float().mean().item())
bad=r.abs()>8*rs
print("bad frac",bad.float().mean().item())
# position range 100..900 only
b=bad[100:900]; print("mid frac",b.float().mean().item())
print("per-row count stats",b.sum(1).float().mean().item(),b.sum(1).float().std().item(),b.sum(1).min().item(),b.sum(1).max().item())
print("per-col count stats",b.sum(0).float().mean().item(),b.sum(0).float().std().item(),b.sum(0).min().item(),b.sum(0).max().item())
xb=x[100:900][b]; pb=pred[100:900][b]
print("examples x, pred, ratio")
for i in range(30): print(f"{xb[i].item():.4f} {pb[i].item():.4f} {xb[i].item()/pb[i].item():.2f}")
print("std bad x",xb.std().item(),"mean",xb.mean().item(),"std pred",pb.std().item(), "corr",np.corrcoef(xb.numpy(),pb.numpy())[0,1])
print("abs quantiles",np.quantile(xb.abs().numpy(),[.01,.1,.5,.9,.99]))
idx=b.nonzero(); print(idx[:40].tolist())
