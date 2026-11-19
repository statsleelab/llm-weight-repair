import torch
torch.set_num_threads(1)
from safetensors.torch import load_file
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=160)
sd = load_file("damaged.safetensors")
a = sd["wpe.weight"].numpy().astype(np.float64)
def rstd(x):
    med = np.median(x); return np.median(np.abs(x-med))*1.4826
cc = np.array([rstd(c) for c in a.T]); small = cc<=0.05
nb = np.full_like(a, np.nan); nb[1:-1]=0.5*(a[:-2]+a[2:])
r = a-nb
m = small[None,:] & ~np.isnan(r) & (np.abs(r)>0.05)
print("spikes in small cols (|res|>0.05):", m.sum(), "of", small.sum()*1022)
v = a[1:-1][m[1:-1]] if False else a[m]
print("spike values hist (abs):", np.histogram(np.abs(v), bins=[0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.5,2,2.5,3,5])[0])
print("sign + / -:", (v>0).sum(), (v<0).sum())
# per-column spike counts
pc = m.sum(0); print("spikes per small column quantiles", np.quantile(pc[small],[0,.5,.9,1]), "mean", pc[small].mean())
print("fraction of small-col entries that are spikes:", m.sum()/ (small.sum()*1022))
for k in ["h.0.attn.c_attn.weight","wte.weight","h.5.mlp.c_proj.weight"]:
    w = sd[k].numpy().astype(np.float64).ravel()
    print(k, "frac |x|>0.5:", (np.abs(w)>0.5).mean().round(4), ">1:", (np.abs(w)>1).mean().round(4), ">2:", (np.abs(w)>2).mean().round(4), ">3:", (np.abs(w)>3).mean().round(5))
