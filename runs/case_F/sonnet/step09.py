from common import *
import sys
ids=devids(4096)
sd=load_weights("damaged.safetensors")
MASS=[138,378,447]
def flag(w,t,absmin):
    med=w.median(0).values
    mad=(w-med).abs().median(0).values*1.4826
    r=(w-med).abs()
    return (r>t*mad.clamp(min=1e-4))&(r>absmin)
def protect(k,m):
    m=m.clone()
    if k=='wte.weight' or 'c_proj' in k: m[:,MASS]=False
    elif 'c_attn' in k or 'c_fc' in k: m[MASS,:]=False
    return m
def fix_wpe(w,t=0.1):
    pad=torch.nn.functional.pad(w.T[None],(2,2),mode='replicate')[0].T
    nb=torch.stack([pad[0:-4],pad[1:-3],pad[3:-1],pad[4:]],0)
    med=nb.median(0).values
    return torch.where((w-med).abs()>t,med,w)
for t,am in [(5,0.4),(4,0.3)]:
    s2=dict(sd)
    s2['wpe.weight']=fix_wpe(sd['wpe.weight'])
    for k,v in sd.items():
        if v.dim()==2 and k!='wpe.weight':
            s2[k]=torch.where(protect(k,flag(v,t,am)),torch.zeros_like(v),v)
    print(t,am,"ppl",perplexity(s2,ids,window=512),flush=True)
