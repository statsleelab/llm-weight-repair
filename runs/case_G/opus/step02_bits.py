import torch; torch.set_num_threads(1)
import numpy as np
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k in ["h.0.attn.c_attn.weight","h.5.mlp.c_fc.weight","h.3.mlp.c_proj.weight","wpe.weight","wte.weight"]:
    v=sd[k].numpy()
    b=v.view(np.uint32)
    low16=(b&0xFFFF)==0; low13=(b&0x1FFF)==0
    print(k,"low16zero",low16.mean(),"low13zero",low13.mean())
    # kurtosis, tails
    s=v.std(); 
    print("  quantiles abs/std",[float(np.quantile(np.abs(v)/s,q)) for q in [0.5,0.9,0.99,0.999]])
    # row/col norm z-scores
    rn=np.linalg.norm(v,axis=1); cn=np.linalg.norm(v,axis=0)
    print("  row norm cv",rn.std()/rn.mean(),"col norm cv",cn.std()/cn.mean())
