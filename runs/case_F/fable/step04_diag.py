import torch, sys, math
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())[:256]
sd=load_weights(sys.argv[1])
lg,h=forward(sd,torch.tensor([[EOT]+ids]),return_hidden=True)
for i,x in enumerate(h): print(i, "rms",x.pow(2).mean().sqrt().item(),"max",x.abs().max().item(), "argmax dim", x.abs().amax((0,1)).argmax().item())
print("logit max",lg.abs().max().item())
for k,v in sd.items():
    if v.ndim==1: print(k, v.abs().max().item(), v.std().item()) if v.abs().max()>20 or not torch.isfinite(v).all() else None
