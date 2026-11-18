import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
def info(k,ks=(0,8,32,128)):
    v=sd[k]
    U,S,Vh=torch.linalg.svd(v,full_matrices=False)
    e=S**2; c=e.cumsum(0)/e.sum()
    out=[]
    for r in ks:
        L=(U[:,:r]*S[:r])@Vh[:r] if r else torch.zeros_like(v)
        E=v-L; mad=E.abs().median()*1.4826
        out.append(f"r{r}: E.std={E.std():.4f} frac>5mad={((E.abs()>5*mad).float().mean()):.4f}")
    print(k,"energy top8/32/128:",[round(c[i-1].item(),3) for i in (8,32,128)],"|"," ".join(out),flush=True)
for k in ["h.0.attn.c_attn.weight","h.4.attn.c_attn.weight","h.8.attn.c_attn.weight","h.4.mlp.c_fc.weight","h.8.mlp.c_fc.weight","h.8.mlp.c_proj.weight","h.8.attn.c_proj.weight","h.0.mlp.c_fc.weight"]:
    info(k)
