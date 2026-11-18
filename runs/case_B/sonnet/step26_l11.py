import torch,numpy as np,math
import torch.nn.functional as F
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("cand_wide.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:400]
x=torch.tensor([[EOT]+ids])
lg,h=forward(sd,x,return_hidden=True)
def lens(r): 
    y=F.layer_norm(r,(768,),sd["ln_f.weight"],sd["ln_f.bias"],1e-5)@sd["wte.weight"].T
    return F.cross_entropy(y[0,5:-1],x[0,6:]).item()
for L in [10,11]:
    p=f"h.{L}."; r=h[L]; B,T,_=r.shape
    hh=F.layer_norm(r,(768,),sd[p+"ln_1.weight"],sd[p+"ln_1.bias"],1e-5)
    q,k,v=(hh@sd[p+"attn.c_attn.weight"]+sd[p+"attn.c_attn.bias"]).split(768,-1)
    q,k,v=[t.view(B,T,12,-1).transpose(1,2) for t in (q,k,v)]
    att=(q@k.transpose(-1,-2))/8; att=att.masked_fill(~torch.tril(torch.ones(T,T,dtype=torch.bool)),float('-inf')).softmax(-1)
    a=(att@v).transpose(1,2).reshape(B,T,768)@sd[p+"attn.c_proj.weight"]+sd[p+"attn.c_proj.bias"]
    r2=r+a
    h2=F.layer_norm(r2,(768,),sd[p+"ln_2.weight"],sd[p+"ln_2.bias"],1e-5)
    m=gelu_new(h2@sd[p+"mlp.c_fc.weight"]+sd[p+"mlp.c_fc.bias"])@sd[p+"mlp.c_proj.weight"]+sd[p+"mlp.c_proj.bias"]
    print(L,"in %.2f after attn %.2f after mlp %.2f"%(lens(r),lens(r2),lens(r2+m)),"rms attn out %.2f mlp out %.2f"%(a[0,1:].pow(2).mean().sqrt(),m[0,1:].pow(2).mean().sqrt()))
