from repairlib import *
ids=devids(2048)
for t in [3.5,5]:
    s2=build(t)
    fr=sum(((s2[k]==0)&(sd[k]!=0)).sum().item() for k in sd if sd[k].dim()==2)/sum(sd[k].numel() for k in sd if sd[k].dim()==2)
    print(t,"zeroed frac",fr,perplexity(s2,ids,window=512),flush=True)
