import torch, numpy as np, math
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
print("dev tokens",len(ids))
for k in ["wpe.weight","wte.weight","h.0.attn.c_attn.weight","h.0.attn.c_proj.weight","h.5.mlp.c_fc.weight","h.5.mlp.c_proj.weight","h.11.mlp.c_proj.weight"]:
    v=sd[k]; m=(v==0)
    r=m.sum(1).float(); c=m.sum(0).float()
    p=m.float().mean().item()
    # expected binomial std
    print(k, "n0",int(m.sum()), "row cnt mean/std/max/min",r.mean().item(),r.std().item(),r.max().item(),r.min().item(),"binom std",math.sqrt(v.shape[1]*p*(1-p)),
          "| col",c.mean().item(),c.std().item(),c.max().item(),c.min().item(),"binom",math.sqrt(v.shape[0]*p*(1-p)))
    # adjacency: fraction of zero whose right neighbour zero
    print("   adj-right",(m[:,1:]&m[:,:-1]).float().sum().item()/m.sum().item(),"adj-down",(m[1:]&m[:-1]).float().sum().item()/m.sum().item())
    # fully zero rows/cols
    print("   full zero rows",int((r==v.shape[1]).sum()),"cols",int((c==v.shape[0]).sum()), "top rows",torch.topk(r,5).indices.tolist(), "top cols",torch.topk(c,5).indices.tolist())
    # neighbours magnitude: are nonzero-entries' dist affected? compare abs mean of entries in rows w/ many zeros
print("ppl damaged 8000:",perplexity(sd,ids,max_tokens=8192))
print(generate(sd,tok,"The capital of France is",15))
