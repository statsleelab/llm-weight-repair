import torch,numpy as np
from gpt2_lib import *
d=load_weights("damaged.safetensors"); c=load_weights("cand_wide.safetensors")
for key in ['h.11.mlp.c_proj.weight','h.11.mlp.c_fc.weight','h.11.attn.c_proj.weight','h.10.mlp.c_proj.weight']:
    dl=(d[key]-c[key]); n=(dl!=0).sum().item()
    top=dl.abs().flatten().topk(8)
    print(key,"changed",n,"energy of change",dl.pow(2).sum().item(),"orig energy",d[key].pow(2).sum().item())
    print("   ",[(divmod(i.item(),d[key].shape[1]),round(d[key].flatten()[i].item(),2),round(c[key].flatten()[i].item(),2)) for i in top.indices])
print(d['ln_f.weight'].abs().max(),d['ln_f.weight'][[138,378,447]],d['ln_f.weight'].mean())
