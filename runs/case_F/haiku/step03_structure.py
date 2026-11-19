import torch
torch.set_num_threads(1)
from safetensors.torch import load_file
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=160)
sd = load_file("damaged.safetensors")
def rstd(x):
    med = np.median(x); return np.median(np.abs(x-med))*1.4826
for k in ["wpe.weight","wte.weight","h.0.attn.c_attn.weight","h.5.mlp.c_proj.weight"]:
    a = sd[k].numpy().astype(np.float64)
    print("==", k, a.shape)
    # histogram of values
    h, e = np.histogram(a.ravel(), bins=30)
    print("hist counts", h)
    print("edges", np.round(e,3))
    # per-row and per-col robust std
    rr = np.array([rstd(r) for r in a]); cc = np.array([rstd(c) for c in a.T])
    print("row rstd quantiles", np.quantile(rr,[0,.05,.5,.95,1]))
    print("col rstd quantiles", np.quantile(cc,[0,.05,.5,.95,1]))
    print("row absmax quantiles", np.quantile(np.abs(a).max(1),[0,.5,.95,1]))
    print("col absmax quantiles", np.quantile(np.abs(a).max(0),[0,.5,.95,1]))
    print("first 8 rows absmax", np.abs(a[:8]).max(1))
