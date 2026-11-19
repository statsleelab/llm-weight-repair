import torch, math
P_C=0.03; SIG_C=10.0
def mad(x,dim):
    m=x.median(dim,keepdim=True).values
    return (x-m).abs().median(dim,keepdim=True).values*1.4826
def gauss_tail(t):  # P(|N(0,1)|>t)
    return torch.erfc(torch.as_tensor(t)/math.sqrt(2))
def wpe_resid(v):
    T=v.shape[0]; nb=[]
    for d in (-2,-1,1,2):
        idx=(torch.arange(T)+d).clamp(0,T-1); nb.append(v[idx])
    pred=torch.stack(nb).median(0).values
    return v-pred, pred
def detect(k, v, T=5.0, protect=True, heavy_mult=4.0):
    gm=((v-v.median()).abs().median()*1.4826).item()
    if k=="wpe.weight":
        r,pred=wpe_resid(v)
        s=torch.maximum(mad(r,0), torch.tensor(gm*0.05))
        z=r/s
        mask=z.abs()>T
        # second pass with cleaned neighbours
        v2=v.clone(); v2[mask]=pred[mask]
        r,pred=wpe_resid(v2); r=v-pred
        z=r/s; mask=z.abs()>T
        return mask, pred, gm
    s=torch.maximum(torch.maximum(mad(v,1),mad(v,0)), torch.tensor(gm))
    z=v/s
    mask=(z.abs()>T)&(v.abs()<60*gm)
    if protect:
        # expected count of flagged corrupted entries per entry ~ P_C * P(|N(0,SIG_C*gm)|>T*s)
        pe=P_C*gauss_tail(T*s/(SIG_C*gm))
        for dim in (0,1):
            cnt=mask.sum(dim,keepdim=True).float(); exp=pe.sum(dim,keepdim=True)
            heavy=cnt>heavy_mult*exp+6
            mask=mask&~heavy
    return mask, torch.zeros_like(v), gm
def repair(sd, T=5.0, protect=True, verbose=False, **kw):
    out={}; stats={}
    for k,v in sd.items():
        if v.dim()!=2: out[k]=v.clone(); continue
        mask,pred,gm=detect(k,v,T,protect,**kw)
        w=v.clone(); w[mask]=pred[mask]; out[k]=w
        stats[k]=mask.float().mean().item()
    if verbose:
        for k,f in stats.items(): print(f"  {k}: {f:.4f}")
    return out
