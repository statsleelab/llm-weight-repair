import torch, sys, math
torch.set_num_threads(1)
from gpt2_lib import *
import torch.nn.functional as F
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())[:256]
sd=load_weights(sys.argv[1]); dm=load_weights("damaged.safetensors")
lg,h=forward(sd,torch.tensor([[EOT]+ids]),return_hidden=True)
x=F.layer_norm(h[-1],(768,),sd["ln_f.weight"],sd["ln_f.bias"])[0]
xm=x.abs().mean(0); top=xm.topk(8)
print("top lnf dims",top.indices.tolist(),[round(v,1) for v in top.values.tolist()])
print("x mean in those", x.mean(0)[top.indices].tolist())
W=sd["wte.weight"]; D=dm["wte.weight"]
for d in top.indices.tolist():
    c=W[:,d]; cd=D[:,d]
    print(d,"lnf w",sd["ln_f.weight"][d].item(),"b",sd["ln_f.bias"][d].item(),"col mean %.3f med %.3f mad-std %.3f std %.3f max|.| %.3f ; damaged std %.3f"%(c.mean(),c.median(),(c-c.median()).abs().median()/0.6745,c.std(),c.abs().max(),cd.std()))
t=tok.encode("MODE")[0]; print("MODE id",t)
contrib=(x.mean(0)*W[t]); tk=contrib.abs().topk(6)
print("MODE contrib dims",tk.indices.tolist(),contrib[tk.indices].tolist(),"W",W[t][tk.indices].tolist(),"D",D[t][tk.indices].tolist())
# per-layer contribution check: h stats per position
print("h12 norm by pos", h[-1][0].norm(dim=-1)[:6].tolist())
print("h1 norm by pos", h[1][0].norm(dim=-1)[:6].tolist(), "h3", h[3][0].norm(dim=-1)[:6].tolist())
