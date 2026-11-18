import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
W = sd["wpe.weight"].numpy()
# smoothness residual along position
from numpy import median
P = W[100:1000]
med = np.stack([W[100+d:1000+d] for d in (-3,-2,-1,1,2,3)]).copy()
m = np.median(med,0)
r = P-m
s = np.median(np.abs(r))*1.4826
print("robust resid std",s)
bad = np.abs(r)>8*s
print("frac bad",bad.mean(), bad.sum())
idx = np.argwhere(bad)
print("rows with bad:",np.bincount(idx[:,0]).max(), "cols max:",np.bincount(idx[:,1]).max(), "expected/row",bad.mean()*768, "per col", bad.mean()*900)
rc=np.bincount(idx[:,0],minlength=900); cc=np.bincount(idx[:,1],minlength=768)
print("row count hist",np.bincount(rc)[:20]); print("col count hist",np.bincount(cc)[:40])
# relation between bad value and expected
x=P[bad]; e=m[bad]
print("corr",np.corrcoef(x,e)[0,1])
for i in range(30): print(idx[i], f"got {x[i]:.4f} exp {e[i]:.4f} ratio {x[i]/e[i]:.3f}")
print("std bad",x.std(),"std exp",e.std(), "mean bad",x.mean())
print("ratio quantiles",np.quantile(x/e,[.05,.25,.5,.75,.95]))
# bits
u = W.view(np.uint32)
print("low byte zero frac all", ((u&0xff)==0).mean(), "bad:", ((P.view(np.uint32)[bad]&0xff)==0).mean())
print("low 13 bits zero frac all", ((u&0x1fff)==0).mean(), "bad:", ((P.copy().view(np.uint32)[bad]&0x1fff)==0).mean())
