import torch,numpy as np
import torch.nn.functional as F
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("cand_heavy.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:300]
x=torch.tensor([[EOT]+ids])
lg=forward(sd,x)[0,:-1]
nll=F.cross_entropy(lg,x[0,1:],reduction='none')
print("median nll",nll.median().item(),"mean",nll.mean().item(),"max",nll.max().item())
print("top1 acc",(lg.argmax(-1)==x[0,1:]).float().mean().item())
print("worst:",[(tok.decode([x[0,i+1]]),round(nll[i].item())) for i in nll.topk(8).indices])
print("pos 0 nll",nll[0].item(), "pos1-5",nll[1:6])
print(generate(sd,tok,"The capital of France is",12))
lp=lg.log_softmax(-1); print("logit std per pos",lg.std(-1)[:10], "lmean",lg.mean(-1)[:10])
