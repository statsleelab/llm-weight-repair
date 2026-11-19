from repairlib import *
import torch.nn.functional as F
s2=build(4,0.3)
ids=devids(512)
x=torch.tensor([[EOT]+ids[:511]])
lg,hs=forward(s2,x,return_hidden=True)
for i,h in enumerate(hs):
    l=F.layer_norm(h,(768,),s2['ln_f.weight'],s2['ln_f.bias'],1e-5)@s2['wte.weight'].T
    nll=F.cross_entropy(l[0,5:-1],x[0,6:],reduction='mean').item()
    acc=(l[0,5:-1].argmax(-1)==x[0,6:]).float().mean().item()
    # self-token: does the current token rank high
    selfacc=(l[0,5:].argmax(-1)==x[0,5:]).float().mean().item()
    print(i,"nll %.2f acc %.3f self-top1 %.3f"%(nll,acc,selfacc))
