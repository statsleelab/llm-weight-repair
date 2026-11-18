import torch, numpy as np, math
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k in ["h.0.attn.c_attn.weight","h.5.mlp.c_proj.weight","wpe.weight","wte.weight"]:
    m=(sd[k]==0)
    r=m.float().sum(1); c=m.float().sum(0)
    print(k,"row cnt mean/std/max",r.mean().item(),r.std().item(),r.max().item(),"poisson std",math.sqrt(r.mean().item()),
          "col cnt mean/std/max",c.mean().item(),c.std().item(),c.max().item(), math.sqrt(c.mean().item()))
    # adjacency
    print("  adjacent pairs frac", (m[:,1:]&m[:,:-1]).float().mean().item())
for k,v in sd.items():
    if v.ndim==1: print(k, (v==0).sum().item(), end="; ")
print()
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read()); print(len(ids))
nll,acc=nll_windows(sd,ids,max_tokens=8192)
print("damaged ppl",math.exp(nll),"acc",acc)
print(generate(sd,tok,"The capital of France is",15))
