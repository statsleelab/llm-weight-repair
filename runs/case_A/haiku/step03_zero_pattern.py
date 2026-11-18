import torch
torch.set_num_threads(1)
from safetensors.torch import load_file
import numpy as np
sd = {k: v.float() for k, v in load_file("/home/user/runs/case_A/haiku/damaged.safetensors").items()}
mats = [k for k in sd if sd[k].dim()==2]
tot_zero = 0; tot = 0
for k in mats:
    Z = (sd[k]==0).numpy()
    tot_zero += Z.sum(); tot += Z.size
    r = Z.sum(1); c = Z.sum(0)
    p = Z.mean()
    # dispersion check: variance/mean of per-row and per-col zero counts (binomial => ~1)
    dr = r.var()/max(r.mean(),1e-9); dc = c.var()/max(c.mean(),1e-9)
    # count of zero entries that are tiny-|x| neighbours? just report
    if k in ("wte.weight","wpe.weight","h.0.attn.c_attn.weight","h.6.mlp.c_proj.weight","ln_f.weight") or k.endswith("c_fc.weight"):
        print(f"{k:28s} frac0={p:.5f} rowmax={r.max()} colmax={c.max()} row_disp={dr:.2f} col_disp={dc:.2f}")
    else:
        print(f"{k:28s} frac0={p:.5f} row_disp={dr:.2f} col_disp={dc:.2f}")
print("overall zero frac over 2D mats:", tot_zero/tot, "count", int(tot_zero))
# Are zeros in wte coincident with zeros in wpe etc? Check whether any row of wte is entirely zero
W = sd["wte.weight"].numpy()
print("wte all-zero rows:", int(((W==0).all(1)).sum()), " wpe all-zero rows:", int(((sd['wpe.weight'].numpy()==0).all(1)).sum()))
