import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
def lr(a,b): return (a.pow(2).mean().sqrt()/ (b.pow(2).mean().sqrt()+1e-9)).item()
for L in [0,1,5,9,11]:
  for nm,ax in [("attn.c_attn",0),("mlp.c_fc",0),("attn.c_proj",1),("mlp.c_proj",1)]:
    w=sd[f"h.{L}.{nm}.weight"]
    if ax==1: w=w.T   # now rows = output dim (768) for c_proj... transpose so d_model axis is 0
    R,C=w.shape
    ch=64
    ins=[];
    for j in range(0,C-ch+1,ch):
        a=w[591:724,j:j+ch]; b=torch.cat([w[400:591,j:j+ch],w[724:768,j:j+ch]])
        ins.append(round(lr(a,b),2))
    ins=np.array(ins)
    print(L,nm,"min %.2f max %.2f"%(ins.min(),ins.max()),"n out of [0.8,1.25]:",((ins<0.8)|(ins>1.25)).sum(),"of",len(ins))
