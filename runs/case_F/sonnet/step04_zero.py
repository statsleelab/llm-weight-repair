from common import *
import sys,time
ids=devids(4096)
sd=load_weights("damaged.safetensors")
t=time.time()
print("damaged ppl",perplexity(sd,ids,window=512),time.time()-t)
def flag(w,t,absmin):
    med=w.median(0).values
    mad=(w-med).abs().median(0).values*1.4826
    r=(w-med).abs()
    return (r>t*mad.clamp(min=1e-4))&(r>absmin)
for t,am in [(6,0.4),(8,0.5)]:
    s2=dict(sd); tot=0;n=0
    for k,v in sd.items():
        if v.dim()==2:
            m=flag(v,t,am); tot+=m.sum().item(); n+=m.numel()
            s2[k]=torch.where(m,torch.zeros_like(v),v)
    print(t,am,"flagged frac",tot/n,"ppl",perplexity(s2,ids,window=512))
