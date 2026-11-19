from common import *
sd=load_weights("damaged.safetensors")
def scale_model(w,iters=4):
    A=w.abs()
    a=torch.ones(w.shape[0],1); b=torch.ones(1,w.shape[1])
    for _ in range(iters):
        b=(A/a).median(0).values[None,:]*1.4826
        a=(A/b).median(1).values[:,None]*1.4826
    return a*b
MASS=[138,378,447]
def legit(k,w,hi=8.0,mid=3.0):
    L=w.abs()>hi
    big=w.abs()>2.5
    if 'c_proj' in k:
        cnt=big[:,MASS].sum(1)
        pair=(cnt>=2)[:,None].expand_as(w).clone()
        line=torch.zeros_like(w,dtype=torch.bool); line[:,MASS]=True
        L|=pair&line&(w.abs()>mid)
    elif 'c_fc' in k or 'c_attn' in k:
        cnt=big[MASS,:].sum(0)
        pair=(cnt>=2)[None,:].expand_as(w).clone()
        line=torch.zeros_like(w,dtype=torch.bool); line[MASS,:]=True
        L|=pair&line&(w.abs()>mid)
    return L
def flag(k,w,t,keep=None):
    S=scale_model(w)
    return (w.abs()>t*S)&~legit(k,w)
def fix_wpe(w,t=0.1):
    pad=torch.nn.functional.pad(w.T[None],(2,2),mode='replicate')[0].T
    nb=torch.stack([pad[0:-4],pad[1:-3],pad[3:-1],pad[4:]],0)
    med=nb.median(0).values
    thr=torch.full((w.shape[0],1),t); thr[:4]=0.35; thr[-3:]=0.25
    # early rows: reference = median over rows 1..6 excluding self
    med=med.clone()
    for r in range(4):
        idx=[j for j in range(0,7) if j!=r]
        med[r]=w[idx].median(0).values
    return torch.where((w-med).abs()>thr,med,w)
def build(t,keep=9.0):
    s2=dict(sd)
    s2['wpe.weight']=fix_wpe(sd['wpe.weight'])
    for k,v in sd.items():
        if v.dim()==2 and k!='wpe.weight':
            s2[k]=torch.where(flag(k,v,t,keep),torch.zeros_like(v),v)
    return s2

import math
SIG=1.24; RATE=0.03
def posterior(k,w,rate=RATE,sig=SIG):
    S=scale_model(w).clamp(min=1e-4)
    z=(w/S).abs()
    N=w.numel()
    zmax=float(z.max())
    nb=80
    edges=torch.linspace(0,math.sqrt(zmax+1e-6),nb+1)**2
    idx=torch.bucketize(z.flatten(),edges[1:-1])
    cnt=torch.bincount(idx,minlength=nb).float()
    wid=edges[1:]-edges[:-1]; cen=(edges[1:]+edges[:-1])/2
    fobs=cnt/(N*wid)
    Ss=S.flatten()[torch.randperm(N)[:20000]]
    fcorr=torch.stack([rate*(2*Ss*torch.exp(-0.5*(c*Ss/sig)**2)/(sig*math.sqrt(2*math.pi))).mean() for c in cen])
    fc=((fobs-fcorr).clamp(min=0)/(1-rate))
    fc=torch.maximum(fc,fobs*0.02)   # floor
    fc_i=fc[idx].view_as(w)
    fcorr_i=rate*2*S*torch.exp(-0.5*(w/sig)**2)/(sig*math.sqrt(2*math.pi))
    P=fcorr_i/((1-rate)*fc_i+fcorr_i)
    return P
def build2(rate=RATE,sig=SIG,hard=None):
    s2=dict(sd)
    s2['wpe.weight']=fix_wpe(sd['wpe.weight'])
    for k,v in sd.items():
        if v.dim()==2 and k!='wpe.weight':
            P=posterior(k,v,rate,sig)
            P=torch.where(legit(k,v),torch.zeros_like(P),P)
            if hard is not None: P=(P>hard).float()
            s2[k]=v*(1-P)
    return s2

def layer_of(k):
    return int(k.split('.')[1]) if k.startswith('h.') else -1
def build3(split=6,t=3.5,hard=0.5,wte_mode='post'):
    s2=dict(sd)
    s2['wpe.weight']=fix_wpe(sd['wpe.weight'])
    for k,v in sd.items():
        if v.dim()==2 and k!='wpe.weight':
            L=layer_of(k)
            if L>=0 and L<split:
                s2[k]=torch.where(flag(k,v,t),torch.zeros_like(v),v)
            else:
                P=posterior(k,v); P=torch.where(legit(k,v),torch.zeros_like(P),P)
                s2[k]=torch.where(P>hard,torch.zeros_like(v),v)
    return s2
