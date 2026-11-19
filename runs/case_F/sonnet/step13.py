from repairlib import *
import torch.nn.functional as F
s2=build(4,0.3)
ids=devids(512)
x=torch.tensor([[EOT]+ids[:511]])
lg,hs=forward(s2,x,return_hidden=True)
nll=F.cross_entropy(lg[0,:-1],x[0,1:],reduction='none')
print("nll first 8",nll[:8].tolist()); print("median",nll.median().item(),"mean",nll.mean().item(),"mean excluding first3",nll[3:].mean().item())
print("logit absmax",lg.abs().max().item())
print("hidden pos0 final",hs[-1][0,0].abs().max().item(), "pos 1", hs[-1][0,1].abs().max().item())
