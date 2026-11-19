import torch
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k in ["h.0.attn.c_proj.weight","h.5.mlp.c_fc.weight","h.3.mlp.c_proj.weight","wpe.weight","wte.weight","h.8.attn.c_attn.weight"]:
    v=sd[k]; med=v.median(); mad=(v-med).abs().median().item()*1.4826
    z=((v-med).abs()/mad)
    print(k,"robust sigma",round(mad,4),"n=",v.numel())
    for t in [4,6,8,10,15,20,30,50]:
        print("  z>",t,(z>t).sum().item(), end=";")
    print()
    top=v.flatten().abs().topk(8)
    print("  top abs",[round(x,2) for x in top.values.tolist()])
    # sign/bit hints: values that are exact powers of two or tiny
    print("  frac |v|<1e-6:",(v.abs()<1e-6).float().mean().item())
