from repairlib import *
import sys
g=sd['ln_f.weight']
ids=devids(2048)
gm=g.median()
def teff(sens,base=3.5,alpha=0.5,tmin=1.2):
    r=(sens/gm).clamp(min=1)
    return (base/(1+alpha*(r-1))).clamp(min=tmin)
def b7(alpha,zerog=3.0,base=3.5):
    s2=dict(sd); s2['wpe.weight']=fix_wpe(sd['wpe.weight'])
    tj=teff(g,base,alpha)
    for k,v in sd.items():
        if v.dim()!=2 or k=='wpe.weight': continue
        S=scale_model(v).clamp(min=1e-4)
        t=torch.full_like(v,base)
        if 'c_proj' in k or k=='wte.weight':
            t=tj[None,:].expand_as(v)
        else:
            L=layer_of(k); ln=sd[f'h.{L}.ln_1.weight'] if 'c_attn' in k else sd[f'h.{L}.ln_2.weight']
            t=(teff(ln,base,alpha)[:,None]).expand_as(v)   # row gain
        m=(v.abs()>t*S)&~legit(k,v)
        w=torch.where(m,torch.zeros_like(v),v)
        if 'c_proj' in k or k=='wte.weight': w[:,g>zerog]=0
        s2[k]=w
    return s2
for alpha,zg in [(0.5,3.0),(1.0,3.0)]:
    print(alpha,zg,perplexity(b7(alpha,zg),ids,window=512),flush=True)
