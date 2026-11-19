from repairlib import *
g=sd['ln_f.weight']; big=(g>5)
print("big dims",big.nonzero().flatten().tolist())
ids=devids(2048)
def b5(zero_wte=False,gmin=5):
    big=(g>gmin)
    s2=build(3.5)
    for k,v in sd.items():
        if v.dim()==2 and 'c_proj' in k:
            w=s2[k].clone(); w[:,big]=0; s2[k]=w
    if zero_wte:
        w=s2['wte.weight'].clone(); w[:,big]=0; s2['wte.weight']=w
    return s2
for zw in [False,True]:
    s2=b5(zw); print("zero_wte",zw,perplexity(s2,ids,window=512),flush=True)
