import torch,numpy as np,math
import torch.nn.functional as F
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("cand_wide.safetensors"); dm=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:400]
x=torch.tensor([[EOT]+ids])
lg,h=forward(sd,x,return_hidden=True)
L=11;p=f"h.{L}.";r=h[L];B,T,_=r.shape
hh=F.layer_norm(r,(768,),sd[p+"ln_1.weight"],sd[p+"ln_1.bias"],1e-5)
print("ln1 out rms",hh[0,1:].pow(2).mean().sqrt().item(),"max",hh[0,1:].abs().max().item(),"pos0 max",hh[0,0].abs().max().item())
print("ln1 weight top",sd[p+"ln_1.weight"].abs().topk(4), "bias top",sd[p+"ln_1.bias"].abs().topk(4))
for name,W in [("cand",sd[p+"attn.c_attn.weight"]),("damaged",dm[p+"attn.c_attn.weight"])]:
    qkv=hh@W+sd[p+"attn.c_attn.bias"]
    print(name,"q/k/v rms",[round(t[0,1:].pow(2).mean().sqrt().item(),2) for t in qkv.split(768,-1)],"pos0 v rms",qkv[0,0,1536:].pow(2).mean().sqrt().item())
print("c_attn bias v rms",sd[p+"attn.c_attn.bias"][1536:].pow(2).mean().sqrt().item())
