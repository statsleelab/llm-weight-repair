import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k in ["wte.weight","wpe.weight","h.0.attn.c_attn.weight","h.5.mlp.c_fc.weight","h.5.mlp.c_proj.weight","h.11.attn.c_proj.weight"]:
    v=sd[k]
    bits=v.numpy().view(np.uint32)
    low=bits & 0xFFFF
    print(k,"low16==0 frac",(low==0).mean(), "low13==0",((bits&0x1FFF)==0).mean())
    # robust z per column and per row
    med_c=v.median(0,keepdim=True).values; mad_c=(v-med_c).abs().median(0,keepdim=True).values
    med_r=v.median(1,keepdim=True).values; mad_r=(v-med_r).abs().median(1,keepdim=True).values
    zc=(v-med_c).abs()/mad_c; zr=(v-med_r).abs()/mad_r
    for t in [5,8,12,20]:
        print(f"  z>{t}: col {(zc>t).float().mean():.5f} row {(zr>t).float().mean():.5f} both {((zc>t)&(zr>t)).float().mean():.5f}")
    # repeated values
    u,c=np.unique(v.numpy(),return_counts=True)
    print("  unique frac",len(u)/v.numel(),"max count",c.max(), "most common", u[c.argmax()])
