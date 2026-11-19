import torch
torch.set_num_threads(1)
from gpt2_lib import *
KEYS=lambda sd:[k for k,v in sd.items() if v.ndim==2]
def zscore(v):
    med=v.median(0,keepdim=True).values; x=v-med
    a=x.abs()
    g=a.median().item()*1.4826
    rm=a.median(1,keepdim=True).values*1.4826
    cm=a.median(0,keepdim=True).values*1.4826
    s=(rm*cm).sqrt().clamp_min(1e-4*g+1e-9)
    return x/s, g, med

def flag(v,T,kc=5.0):
    z,_,med=zscore(v)
    m=z.abs()>T
    n=m.sum().item()
    out=m.clone()
    for dim in (0,1):
        c=m.sum(dim).float()   # dim0 sum -> per column count
        mean=c.mean().item()
        bad=c>(mean+kc*mean**0.5+3)
        if dim==0: out&=~bad[None,:]
        else: out&=~bad[:,None]
    return out,m,med
