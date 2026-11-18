import torch,numpy as np,math
import torch.nn.functional as F
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:200]
x=torch.tensor([[EOT]+ids])
def sink(sd,Ls=(1,3,6,9)):
    lg,h=forward(sd,x,return_hidden=True); out=[]
    for L in Ls:
        p=f"h.{L}.";r=h[L];B,T,_=r.shape
        hh=F.layer_norm(r,(768,),sd[p+"ln_1.weight"],sd[p+"ln_1.bias"],1e-5)
        q,k,v=(hh@sd[p+"attn.c_attn.weight"]+sd[p+"attn.c_attn.bias"]).split(768,-1)
        q,k=[t.view(B,T,12,-1).transpose(1,2) for t in (q,k)]
        att=(q@k.transpose(-1,-2))/8; att=att.masked_fill(~torch.tril(torch.ones(T,T,dtype=torch.bool)),float('-inf')).softmax(-1)
        out.append(round(att[0,:,20:,0].mean().item(),3))
    return out
for f in ["damaged","cand_soft","cand_heavy","cand_wide"]:
    print(f,sink(load_weights(f+".safetensors")))
