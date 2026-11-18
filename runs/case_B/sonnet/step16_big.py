import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
from step15_soft import *
for key in ['h.1.mlp.c_proj.weight','h.2.mlp.c_proj.weight','h.3.mlp.c_proj.weight','h.2.mlp.c_fc.weight','h.1.mlp.c_fc.weight']:
    v=sd[key]; s=scales(v); b,a0=fit(v,s); a=a0*s
    z=(v.abs()/a)
    top=torch.topk(z.flatten(),12)
    print(key,"b",b,"a0",a0)
    for zz,i in zip(top.values,top.indices):
        r,c=divmod(i.item(),v.shape[1]); print("   (%d,%d) w=%.2f z=%.1f"%(r,c,v[r,c],zz))
