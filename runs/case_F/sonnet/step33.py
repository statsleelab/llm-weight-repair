from repairlib import *
g=sd['ln_f.weight']
print("gain>5 dims",(g>5).sum().item(),"gain>10",(g>10).sum().item(),"<0.5",(g<0.5).sum().item())
ids=devids(2048)
big=(g>5)
def b4(tstrong,t=3.5):
    s2=build(t)
    for k,v in sd.items():
        if v.dim()==2 and 'c_proj' in k:
            S=scale_model(v).clamp(min=1e-4)
            m=(v.abs()>tstrong*S)&~legit(k,v); m[:,~big]=False
            s2[k]=torch.where(m|(s2[k]==0),torch.zeros_like(v),v)
    return s2
for ts in [2.0,1.0]:
    s2=b4(ts)
    print(ts,perplexity(s2,ids,window=512),flush=True)
