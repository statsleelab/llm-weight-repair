import torch, numpy as np, math
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors"); eb=load_weights("cand_eb1.safetensors")
for k,W in sd.items():
    if W.ndim!=2: continue
    sn=10*W.std().item()/math.sqrt(2)
    N=W.numel()
    A=W.abs()
    n4=(A>4*sn).sum().item(); n5=(A>5.5*sn).sum().item()
    exp4=0.01*6.33e-5*N
    M=A>4*sn
    kept=((eb[k].abs()>0.5*A)&M).sum().item()
    idx=M.nonzero()
    cols=torch.unique(idx[:,1]).tolist()[:8]; rows=torch.unique(idx[:,0]).tolist()[:8]
    print(f"{k:26s} sn={sn:.3f} n>4sn={n4} (exp corrupt {exp4:.1f}) n>5.5sn={n5} kept_by_eb1={kept} rows{rows} cols{cols}")
