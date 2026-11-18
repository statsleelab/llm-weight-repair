import torch, math; torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())[:256]
x=torch.tensor([[EOT]+ids])
logits,hid=forward(sd,x,return_hidden=True)
for i,h in enumerate(hid): print(i, "norm mean", h.norm(dim=-1).mean().item(), "max", h.abs().max().item())
print("logits std", logits.std().item(), logits.abs().max().item())
print(generate(sd,tok,"The capital of France is",10))
# LN stats
for k,v in sd.items():
    if v.dim()==1: print(k, v.shape[0], round(v.mean().item(),3), round(v.abs().max().item(),3))
