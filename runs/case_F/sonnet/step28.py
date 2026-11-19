from repairlib import *
ids=devids(2048)
s2=build2()
tot=sum((s2[k]!=sd[k]).sum().item() for k in sd if sd[k].dim()==2)/sum(sd[k].numel() for k in sd if sd[k].dim()==2)
print("changed frac",tot)
print("soft",perplexity(s2,ids,window=512),flush=True)
s3=build2(hard=0.5)
print("hard0.5",perplexity(s3,ids,window=512),flush=True)
