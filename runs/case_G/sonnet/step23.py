import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
def kmap(w,th,tw):
    R,C=w.shape
    t=w.reshape(R//th,th,C//tw,tw).permute(0,2,1,3).reshape(R//th,C//tw,-1)
    m=t.mean(-1,keepdim=True); v=((t-m)**2).mean(-1)
    k=((t-m)**4).mean(-1)/(v**2+1e-12)
    return k,v.sqrt()
for key in ["h.5.attn.c_attn.weight","h.5.attn.c_proj.weight"]:
    w=sd[key]; g=w.std().item()
    k,s=kmap(w,16,16)
    print(key,"global std",g,"kurt median",k.median().item())
    for i in range(0,k.shape[0]):
        print("".join("o" if (kk<3.4 and abs(ss/g-1)<0.35) else "." for kk,ss in zip(k[i].tolist(),s[i].tolist()))[:150])
