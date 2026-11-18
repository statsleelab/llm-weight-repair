import torch, numpy as np, sys
torch.set_num_threads(1)
from gpt2_lib import *
def local_scale(w):
    a=w.abs()
    c=a.median(0,keepdim=True).values/0.6745
    r=a.median(1,keepdim=True).values/0.6745
    return torch.maximum(c.expand_as(w),r.expand_as(w))
def clip_repair(w,z):
    s=local_scale(w)
    f=(w.abs()>z*s)
    return torch.where(f,torch.zeros_like(w),w),f
if __name__=="__main__":
    sd=load_weights("damaged.safetensors"); tok=Tokenizer("tokenizer")
    ids=tok.encode(open("dev.txt").read())
    groups={"wte":["wte.weight"],"c_attn":[f"h.{i}.attn.c_attn.weight" for i in range(12)],
    "attn_proj":[f"h.{i}.attn.c_proj.weight" for i in range(12)],"c_fc":[f"h.{i}.mlp.c_fc.weight" for i in range(12)],
    "mlp_proj":[f"h.{i}.mlp.c_proj.weight" for i in range(12)]}
    for z in (4.0,6.0):
      for g,ks in groups.items():
        sd2=dict(sd); n=0
        for k in ks:
            sd2[k],f=clip_repair(sd[k],z); n+=f.sum().item()
        print("z",z,g,"flagged",n,"ppl",round(perplexity(sd2,ids,max_tokens=2048),3),flush=True)
