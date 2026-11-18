import torch, numpy as np, math
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, perplexity
a=load_weights("cand_eb4.safetensors"); b=load_weights("cand_eb6.safetensors"); sd=load_weights("damaged.safetensors")
D=(b["wte.weight"]-a["wte.weight"])
big=(D.abs()>0.8)
cc=big.sum(0); top=torch.topk(cc,12)
print("n |D|>0.8:",big.sum().item()," cols:",[(i.item(),int(cc[i])) for i in top.indices])
e=(D**2).sum(0); top=torch.topk(e,12); print("col energy top:",[(i.item(),round(e[i].item(),1)) for i in top.indices], "total",e.sum().item())
W=sd["wte.weight"]
cs=(W-W.median(0).values).abs().median(0).values*1.4826
print("columns with tiny robust scale:",[(i,round(cs[i].item(),4)) for i in torch.argsort(cs)[:12].tolist()])
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
cols=[i.item() for i in torch.topk(cc,3).indices]
m=dict(a); w=a["wte.weight"].clone(); w[:,cols]=b["wte.weight"][:,cols]; m["wte.weight"]=w
print("eb4 + eb6 wte cols",cols,":",perplexity(m,ids,max_tokens=2048))
w=b["wte.weight"].clone(); w[:,cols]=a["wte.weight"][:,cols]; m["wte.weight"]=w
print("eb6 wte except cols",cols,":",perplexity(m,ids,max_tokens=2048))
