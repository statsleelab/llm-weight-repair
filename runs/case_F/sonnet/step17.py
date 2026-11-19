from repairlib import *
ids=devids(2048)
for t,am in [(3.5,0.0),(5,0.0)]:
    s2=build(t,am)
    fr=sum(((s2[k]==0)&(sd[k]!=0)).sum().item() for k in sd if sd[k].dim()==2)/sum(sd[k].numel() for k in sd if sd[k].dim()==2)
    print(t,am,"zeroed frac",fr,perplexity(s2,ids,window=512),flush=True)
