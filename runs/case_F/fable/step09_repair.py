# Empirical-Bayes outlier removal, v3: robust baseline (median polish [+low rank]),
# rank-1 local scale, and a local "genuine heavy tail" rate per row/column so that
# rows/cols whose large entries are real (clustered) are not wiped.
# usage: python step09_repair.py MODE RANK OUT [KEEP=4.5] [EPSMODE=1]
import torch, sys, math
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights
from statistics import NormalDist
MODE=sys.argv[1]; RANK=int(sys.argv[2]); OUT=sys.argv[3]
KEEP=float(sys.argv[4]) if len(sys.argv)>4 else 4.5
EPSMODE=int(sys.argv[5]) if len(sys.argv)>5 else 1
PI=0.03
sd = load_weights("damaged.safetensors")
def scales(R):
    r=torch.ones(R.shape[0],1)
    for _ in range(3):
        c=((R/r).abs().median(0,keepdim=True).values/0.6745); c=c.clamp_min(1e-3*c.median()+1e-12)
        r=((R/c).abs().median(1,keepdim=True).values/0.6745); r=r.clamp_min(1e-3*r.median()+1e-12)
    return r*c
def post(X,M,sn):
    R=X-M; S=scales(R)
    logf=-0.5*(R/S)**2-0.5*math.log(2*math.pi)-torch.log(S)
    v=sn*sn+S*S
    logg=-0.5*R*R/v-0.5*torch.log(2*math.pi*v)
    lo=math.log(PI/(1-PI))+logg-logf
    p=torch.sigmoid(lo)
    if EPSMODE:
        fl=(p>0.5).float()
        eps=torch.zeros_like(R)
        for ax in (1,0):
            f=fl.mean(ax,keepdim=True); n=R.shape[ax]
            base=f.median().item(); sdv=math.sqrt(base*(1-base)/n)
            e=(f-base-2*sdv).clamp_min(0)
            eps=torch.maximum(eps,e.expand_as(eps))
        # genuine density: (1-eps) narrow + eps * broad (same width as noise)
        f_mix=torch.logaddexp(torch.log1p(-eps)+logf, torch.log(eps+1e-30)+logg)
        p=torch.sigmoid(math.log(PI/(1-PI))+logg-f_mix)
    p=torch.where(R.abs()>KEEP*sn,torch.zeros_like(p),p)
    return p,R
def repair(X,sn):
    cm=X.median(0,keepdim=True).values
    rm=(X-cm).median(1,keepdim=True).values
    cm=cm+(X-cm-rm).median(0,keepdim=True).values
    M=cm+rm
    p,R=post(X,M,sn)
    for it in range(3 if RANK>0 else 0):
        Y=torch.where(p>0.5,M,X)-(cm+rm)
        U,s,V=torch.svd_lowrank(Y,q=RANK+8,niter=2)
        M=cm+rm+(U[:,:RANK]*s[:RANK])@V[:,:RANK].T
        p,R=post(X,M,sn)
    Y = X-p*R if MODE=="soft" else torch.where(p>0.5,M,X)
    return Y,p
out={}
for k,v in sd.items():
    if v.ndim!=2: out[k]=v; continue
    a=v.abs().flatten()
    T=torch.quantile(a[::max(1,a.numel()//2000000)].double(),0.995).item()
    sn=T/NormalDist().inv_cdf(1-0.005/PI/2)
    if "c_attn" in k:
        parts=[repair(b,sn) for b in v.split(768,dim=1)]
        Y=torch.cat([q[0] for q in parts],1); p=torch.cat([q[1] for q in parts],1)
    else: Y,p=repair(v,sn)
    out[k]=Y.float()
    print(f"{k:26s} sn={sn:.3f} flagged={(p>0.5).float().mean().item():.4f} sum_p={p.mean().item():.4f}",flush=True)
save_weights(out,OUT)
