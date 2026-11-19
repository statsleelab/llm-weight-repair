import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k in ["h.0.attn.c_attn.weight","h.3.mlp.c_fc.weight","h.3.attn.c_proj.weight","h.8.mlp.c_proj.weight","wte.weight"]:
    a=sd[k].flatten()
    q=torch.quantile(a[:1000000],torch.tensor([.001,.01,.1,.5,.9,.99,.999]))
    print(k,[round(x,3) for x in q.tolist()])
    kurt=((a-a.mean())**4).mean()/a.var()**2
    print(" kurt",kurt.item())
    # bits: low mantissa bits
    u=a.view(torch.int32)
    print(" lowbits zero frac",((u&0xFF)==0).float().mean().item(), "exp hist", torch.bincount(((u>>23)&0xFF)[:2000000].long()).nonzero().flatten()[[0,-1]].tolist())
# neighbor-residual outliers in c_attn h.3: value vs median of 3x3 neighbourhood? weights not smooth. Check column/row stats instead
w=sd["h.3.mlp.c_fc.weight"]
r=(w.abs()>0).float()
print("row max abs top", w.abs().max(1).values.topk(5))
print("col absmean spread",w.abs().mean(0).std().item(),w.abs().mean(0).mean().item())
