from common import *
import torch.nn.functional as F
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:100]
x=torch.tensor([[EOT]+ids])
T=x.shape[1]
e=sd["wte.weight"][x]+sd["wpe.weight"][:T]
print("emb rms",e.pow(2).mean().sqrt().item())
p="h.0."
h=F.layer_norm(e,(768,),sd[p+"ln_1.weight"],sd[p+"ln_1.bias"],1e-5)
print("ln1 absmax",h.abs().max().item(), "ln1 w stats",sd[p+"ln_1.weight"].abs().max().item())
qkv=h@sd[p+"attn.c_attn.weight"]+sd[p+"attn.c_attn.bias"]
print("qkv absmax q,k,v",[t.abs().max().item() for t in qkv.split(768,-1)])
W=sd[p+"attn.c_proj.weight"]; b=sd[p+"attn.c_proj.bias"]
print("attn proj bias 447",b[447].item(), "bias absmax",b.abs().max().item())
m=sd[p+"mlp.c_proj.bias"]; print("mlp proj bias 447",m[447].item())
Wm=sd[p+"mlp.c_proj.weight"]
col=Wm[:,447]; print("mlp c_proj col447 top",col.abs().topk(6))
col=W[:,447]; print("attn c_proj col447 top",col.abs().topk(6))
