import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k in ["h.8.attn.c_attn.weight","h.4.mlp.c_fc.weight","wte.weight","h.8.mlp.c_proj.weight"]:
    v=sd[k]; a=v.numpy().ravel()
    u,c=np.unique(a,return_counts=True)
    print(k,"unique",len(u),"of",a.size,"max dup",c.max(), "top dup vals",u[c.argsort()[-3:]])
    # robust scale outer-product z
    cs=v.abs().median(0).values; rs=v.abs().median(1).values
    z=v/(cs[None,:]*rs[:,None]/rs.median()+1e-9)/1.4826*0.6745*1.0
    zz=(v/(cs[None,:]*rs[:,None]/rs.median()))/ (1/0.6745)
    for t in (3,4,5,6):
        print("  tail z>",t,(zz.abs()>t).float().mean().item())
    # exponent histogram: distribution of log2|v|
    e=torch.log2(v.abs()+1e-30).floor().long().flatten()
    print("  exp hist",{int(i):int((e==i).sum()) for i in range(-30,3) if (e==i).sum()>0})
    # sign balance of large entries per sector
    print("  frac neg",(v<0).float().mean().item())
