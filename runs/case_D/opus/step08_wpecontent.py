import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
w=sd["wpe.weight"]
r0,r1,c0,c1=151,252,59,136
B=w[r0:r1,c0:c1]
a=w[r0-1,c0:c1]; b=w[r1,c0:c1]
t=torch.linspace(0,1,r1-r0+2)[1:-1,None]
interp=a*(1-t)+b*t
print("block mean/std",B.mean().item(),B.std().item(),"interp std",interp.std().item())
print("surround std (rows 100-150, same cols)",w[100:150,c0:c1].std().item(), "(rows 252-300)",w[252:300,c0:c1].std().item())
print("corr block vs interp",np.corrcoef(B.flatten(),interp.flatten())[0,1])
# per-column mean in block vs interp
print("corr col means",np.corrcoef(B.mean(0),interp.mean(0))[0,1])
print("corr row-wise smoothness: lag-1 autocorr along pos in block", np.corrcoef(B[1:].flatten(),B[:-1].flatten())[0,1])
print("lag-1 along cols", np.corrcoef(B[:,1:].flatten(),B[:,:-1].flatten())[0,1])
print("kurtosis", (((B-B.mean())/B.std())**4).mean().item())
# Is the block a copy from elsewhere in the matrix? search for matching
flat=B.flatten()
for k,v in sd.items():
    if v.dim()!=2: continue
    vs=set(np.round(v.numpy().flatten()[:2000000],7).tolist()) if False else None
# check values presence anywhere in wpe outside block
outside=w.clone(); outside[r0:r1,c0:c1]=float('nan')
S=set(outside.numpy().flatten().tolist())
print("frac of block values found elsewhere in wpe", np.mean([x in S for x in flat.tolist()]))
