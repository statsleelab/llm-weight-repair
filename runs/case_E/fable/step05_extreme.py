import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import load_weights
from repair_lib import *
from collections import Counter
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.ndim!=2 or k=="wpe.weight": continue
    g=(1.4826*v.abs().median()).item()
    m,_=detect_generic(v,6.0)
    big = m & (v.abs()>35*g)
    # flagged-per-row / col concentration
    rc=m.sum(1); cc=m.sum(0)
    exp_r = m.float().mean().item()*v.shape[1]; exp_c=m.float().mean().item()*v.shape[0]
    hot_r=(rc>exp_r*4+6).nonzero().flatten().tolist(); hot_c=(cc>exp_c*4+6).nonzero().flatten().tolist()
    if big.sum()>0 or hot_r or hot_c:
        ij=big.nonzero().tolist()
        print(k, "g=%.3f"%g, "n>35g:",len(ij), "rows",Counter(i for i,j in ij).most_common(4),"cols",Counter(j for i,j in ij).most_common(4))
        print("    hot rows",[(i,int(rc[i])) for i in hot_r][:8],"hot cols",[(j,int(cc[j])) for j in hot_c][:8], "expected/row %.1f /col %.1f"%(exp_r,exp_c))
