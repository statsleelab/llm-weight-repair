import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k in ["h.11.attn.c_proj.weight","h.4.attn.c_attn.weight","h.1.attn.c_proj.weight","h.3.mlp.c_proj.weight"]:
    w=sd[k]
    t=w.reshape(w.shape[0]//16,16,w.shape[1]//16,16)
    s=t.abs().mean((1,3))
    z=s-s.median(1,keepdim=True).values-s.median(0,keepdim=True).values+s.median()
    zz=(z-z.median())/((z-z.median()).abs().median()*1.4826)
    ii=(zz.abs()>6).nonzero()
    print(k,"n",len(ii),"rows",ii[:,0].min().item()*16,ii[:,0].max().item()*16+15,"cols",ii[:,1].min().item()*16,ii[:,1].max().item()*16+15)
    print(" row hist",torch.bincount(ii[:,0]).nonzero().flatten().tolist()[:40])
    print(" col hist",torch.bincount(ii[:,1]).nonzero().flatten().tolist()[:40])
