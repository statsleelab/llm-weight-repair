from repairlib import *
import torch.nn.functional as F
s2=build(3.5)
ids=devids(512)
x=torch.tensor([[EOT]+ids[:511]])
lg,hs=forward(s2,x,return_hidden=True)
nll=F.cross_entropy(lg[0,:-1],x[0,1:],reduction='none')
print("nll first 8",[round(v,1) for v in nll[:8].tolist()],"median",nll.median().item(),"mean",nll.mean().item())
print("logit absmax per pos",lg[0].abs().amax(-1)[:8].tolist(), lg[0,100].abs().max().item())
print("final hidden max pos 100", hs[-1][0,100].abs().topk(4))
print(torch.cat([hs[i][0,100:101].abs().amax(-1) for i in range(13)]))
