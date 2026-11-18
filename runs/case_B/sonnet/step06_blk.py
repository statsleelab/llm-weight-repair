import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
import torch.nn.functional as F
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:128]
sd=load_weights("damaged.safetensors")
x=torch.tensor([[EOT]+ids])
T=x.shape[1]
e=sd["wte.weight"][x]+sd["wpe.weight"][:T]
print("emb rms",e.pow(2).mean().sqrt().item(), "wte part",sd["wte.weight"][x].pow(2).mean().sqrt().item(),"wpe part",sd["wpe.weight"][:T].pow(2).mean().sqrt().item())
p="h.0."
h=F.layer_norm(e,(768,),sd[p+"ln_1.weight"],sd[p+"ln_1.bias"],1e-5)
print("ln1 w rms",sd[p+"ln_1.weight"].abs().mean().item())
qkv=h@sd[p+"attn.c_attn.weight"]+sd[p+"attn.c_attn.bias"]
print("qkv rms q,k,v",[t.pow(2).mean().sqrt().item() for t in qkv.split(768,-1)])
W=sd[p+"attn.c_attn.weight"]
print("c_attn col rms (q,k,v)",[W[:,i*768:(i+1)*768].pow(2).mean().sqrt().item() for i in range(3)])
print("row rms top",W.pow(2).mean(1).sqrt().topk(5))
# wpe row norms
w=sd["wpe.weight"]; print("wpe row rms first 10",w.pow(2).mean(1).sqrt()[:10], "median",w.pow(2).mean(1).sqrt().median())
