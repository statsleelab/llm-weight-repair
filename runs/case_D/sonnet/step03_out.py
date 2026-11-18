import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k in ["h.3.mlp.c_proj.weight","h.2.mlp.c_fc.weight","h.7.mlp.c_proj.weight","wte.weight","h.8.attn.c_attn.weight"]:
    v=sd[k]; med=v.median(); mad=(v-med).abs().median()*1.4826
    idx=((v-med).abs()>8*mad).nonzero()
    print(k, v.shape, len(idx))
    r=torch.bincount(idx[:,0],minlength=v.shape[0]); c=torch.bincount(idx[:,1],minlength=v.shape[1])
    print(" top rows",r.topk(4),"top cols",c.topk(4))
    vals=v[idx[:,0],idx[:,1]]
    print(" sample vals",vals[:12])
    # histogram of |v|/mad
    z=((v-med).abs()/mad).flatten()
    print(" quantiles of z",[round(torch.quantile(z[:5000000] if z.numel()>5e6 else z,q).item(),2) for q in (0.5,0.9,0.99,0.999,0.9999)])
