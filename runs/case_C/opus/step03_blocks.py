import torch
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()!=2: continue
    z=(v==0)
    r=z.sum(1); c=z.sum(0)
    # rows/cols participating in block: those with count equal to max
    R=(r>=r.max()*0.5).nonzero()[:,0]; C=(c>=c.max()*0.5).nonzero()[:,0]
    sub=z[R][:,C].float().mean().item()
    outside=z.sum().item()-z[R][:,C].sum().item()
    print(f"{k:26s} R[{R.min().item()}..{R.max().item()}] n={len(R)} contig={R.max().item()-R.min().item()+1==len(R)}  C[{C.min().item()}..{C.max().item()}] n={len(C)} contig={C.max().item()-C.min().item()+1==len(C)} fill={sub:.4f} zeros_outside={outside}")
