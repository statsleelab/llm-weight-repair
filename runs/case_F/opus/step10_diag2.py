import torch
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
import torch.nn.functional as F
ids=torch.load("dev_ids.pt").tolist()
sd=load_weights("damaged.safetensors")
r=repair(sd,T=4,protect=True)
x=torch.tensor([[EOT]+ids[:255]])
lg,hid=forward(r,x,return_hidden=True)
nll=F.cross_entropy(lg[0,:-1],x[0,1:],reduction='none')
print("nll quantiles",torch.quantile(nll,torch.tensor([.1,.5,.9,.99])), nll.max())
print("logit std per pos",lg[0].std(-1)[:10], "mean",lg[0].mean(-1)[:10])
for i,h in enumerate(hid): print(i, "norm pos1..",h[0,1:].norm(dim=-1).mean().item(), "max abs dim", h[0,1:].abs().mean(0).topk(3))
tok=Tokenizer("tokenizer")
print(generate(r,tok,"The capital of France is",15))
