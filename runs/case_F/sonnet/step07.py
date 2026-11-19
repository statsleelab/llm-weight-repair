from common import *
ids=devids(4096)
sd=load_weights("damaged.safetensors")
def fix_wpe(w,t=0.1):
    pad=torch.nn.functional.pad(w.T[None],(2,2),mode='replicate')[0].T
    nb=torch.stack([pad[0:-4],pad[1:-3],pad[3:-1],pad[4:]],0)
    med=nb.median(0).values
    m=(w-med).abs()>t
    return torch.where(m,med,w),m
def flag(w,t,absmin):
    med=w.median(0).values
    mad=(w-med).abs().median(0).values*1.4826
    r=(w-med).abs()
    return (r>t*mad.clamp(min=1e-4))&(r>absmin)
s2=dict(sd)
s2['wpe.weight'],m=fix_wpe(sd['wpe.weight'])
print("wpe only ppl",perplexity(s2,ids,window=512))
for k,v in sd.items():
    if v.dim()==2 and k!='wpe.weight':
        s2[k]=torch.where(flag(v,5,0.45),torch.zeros_like(v),v)
print("all ppl",perplexity(s2,ids,window=512))
