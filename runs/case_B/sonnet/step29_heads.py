import torch,numpy as np,math
import torch.nn.functional as F
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("cand_wide.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:300]
x=torch.tensor([[EOT]+ids])
lg,h=forward(sd,x,return_hidden=True)
for L in [3,11]:
    p=f"h.{L}.";r=h[L];B,T,_=r.shape
    hh=F.layer_norm(r,(768,),sd[p+"ln_1.weight"],sd[p+"ln_1.bias"],1e-5)
    q,k,v=(hh@sd[p+"attn.c_attn.weight"]+sd[p+"attn.c_attn.bias"]).split(768,-1)
    q,k,v=[t.view(B,T,12,-1).transpose(1,2) for t in (q,k,v)]
    att=(q@k.transpose(-1,-2))/8; att=att.masked_fill(~torch.tril(torch.ones(T,T,dtype=torch.bool)),float('-inf')).softmax(-1)
    o=att@v  # B,12,T,64
    Wp=sd[p+"attn.c_proj.weight"]
    print("layer",L)
    for hd in range(12):
        out=o[0,hd,1:]@Wp[hd*64:(hd+1)*64]
        print(" head",hd,"sink attn %.2f"%att[0,hd,20:,0].mean().item(),"o rms %.2f"%o[0,hd,1:].pow(2).mean().sqrt().item(),"out rms %.2f"%out.pow(2).mean().sqrt().item(),"v pos0 rms %.2f"%v[0,hd,0].pow(2).mean().sqrt().item(),"qk max %.1f"%((q[0,hd]@k[0,hd].T)/8).abs().max().item())
