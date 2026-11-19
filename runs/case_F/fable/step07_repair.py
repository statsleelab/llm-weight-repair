# Empirical-Bayes outlier removal around a robust baseline (median polish [+ low rank]).
# usage: python step07_repair.py NU MODE RANK OUT
import torch, numpy as np, sys, math
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights
from statistics import NormalDist
NU=float(sys.argv[1]); MODE=sys.argv[2]; RANK=int(sys.argv[3]); OUT=sys.argv[4]; PI=0.03
sd = load_weights("damaged.safetensors")
def tlogpdf(z,nu):
    if nu>=1e3: return -0.5*z*z-0.5*math.log(2*math.pi)
    c=math.lgamma((nu+1)/2)-math.lgamma(nu/2)-0.5*math.log(nu*math.pi)
    return c-(nu+1)/2*torch.log1p(z*z/nu)
def scales(R):
    r=torch.ones(R.shape[0],1)
    for _ in range(3):
        c=((R/r).abs().median(0,keepdim=True).values/0.6745); c=c.clamp_min(1e-3*c.median()+1e-12)
        r=((R/c).abs().median(1,keepdim=True).values/0.6745); r=r.clamp_min(1e-3*r.median()+1e-12)
    return r*c
def post(X,M,sn):
    R=X-M; S=scales(R)
    logf=tlogpdf(R/S,NU)-torch.log(S)
    # corrupted entry: x = w + n  ->  resid ~ N(0, sn^2 + S^2)
    v=sn*sn+S*S
    logg=-0.5*R*R/v-0.5*torch.log(2*math.pi*v)
    return torch.sigmoid(math.log(PI/(1-PI))+logg-logf), R
out={}
for k,v in sd.items():
    if v.ndim!=2: out[k]=v; continue
    X=v
    a=X.abs().flatten()
    T=torch.quantile(a[::max(1,a.numel()//2000000)].double(),0.995).item()
    sn=T/NormalDist().inv_cdf(1-0.005/PI/2)
    # median polish baseline
    cm=X.median(0,keepdim=True).values
    rm=(X-cm).median(1,keepdim=True).values
    cm=cm+(X-cm-rm).median(0,keepdim=True).values
    M=cm+rm
    p,R=post(X,M,sn)
    for it in range(3 if RANK>0 else 0):
        Y=torch.where(p>0.5,M,X)-(cm+rm)
        U,s,V=torch.svd_lowrank(Y,q=RANK+8,niter=3)
        M=cm+rm+(U[:,:RANK]*s[:RANK])@V[:,:RANK].T
        p,R=post(X,M,sn)
    if MODE=="soft": Y=X-p*R
    else: Y=torch.where(p>0.5,M,X)
    out[k]=Y.float()
    print(f"{k:26s} sn={sn:.3f} flagged={(p>0.5).float().mean().item():.4f} sum_p={p.mean().item():.4f} resid_rs={(R.abs().median()/0.6745).item():.4f} vs raw {(X.abs().median()/0.6745).item():.4f}",flush=True)
save_weights(out,OUT)
