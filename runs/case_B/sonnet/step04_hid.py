import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:256]
sd=load_weights("damaged.safetensors")
x=torch.tensor([[EOT]+ids])
lg,h=forward(sd,x,return_hidden=True)
for i,s in enumerate(h): print(i,"rms %.1f max %.1f"%(s.pow(2).mean().sqrt(),s.abs().max()), "pos0 max %.1f"%s[0,0].abs().max())
# real GPT-2 wte rows of common tokens
w=sd['wte.weight']; print("row norm quantiles",torch.quantile(w[:20000].norm(dim=1),torch.tensor([.01,.5,.99,1.])))
# per-column outlier frac in wte
print((w.abs()>1).float().mean(0).topk(8))
