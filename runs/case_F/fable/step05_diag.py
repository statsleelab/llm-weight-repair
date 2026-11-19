import torch, sys, math
torch.set_num_threads(1)
from gpt2_lib import *
import torch.nn.functional as F
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())[:256]
sd=load_weights(sys.argv[1])
for k,v in sd.items():
    if v.ndim==1 and ('ln' in k): print(k, "mean %.3f std %.3f min %.3f max %.3f"%(v.mean(),v.std(),v.min(),v.max()))
lg,h=forward(sd,torch.tensor([[EOT]+ids]),return_hidden=True)
x=F.layer_norm(h[-1],(768,),sd["ln_f.weight"],sd["ln_f.bias"])
print("lnf out rms",x.pow(2).mean().sqrt().item(), x.abs().max().item())
top=lg[0].argmax(-1)
print(repr(tok.decode(top[:40])))
vals,cnt=torch.unique(top,return_counts=True)
print([(repr(tok.decode([v])),c.item()) for v,c in zip(vals,cnt) if c>5])
W=sd["wte.weight"]; n=W.norm(dim=1)
print("wte row norms: median",n.median().item(),"max",n.max().item(),"top ids",n.topk(10).indices.tolist(), n.topk(10).values.tolist())
print("lg rows max mean", lg[0].max(-1).values.mean().item(), "target logit mean", lg[0,torch.arange(256),torch.tensor(ids)].mean().item())
