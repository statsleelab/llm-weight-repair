# Empirical-Bayes outlier removal. usage: python step03_repair.py NU MODE OUT
import torch, numpy as np, sys, math
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights
from statistics import NormalDist
NU=float(sys.argv[1]); MODE=sys.argv[2]; OUT=sys.argv[3]; PI=0.03
sd = load_weights("damaged.safetensors")
def tlogpdf(z,nu):
    if nu>=1e3: return -0.5*z*z-0.5*math.log(2*math.pi)
    # unit-MAD-matched scale is close enough; plain t with scale adj so median|z| matches normal's
    c=math.lgamma((nu+1)/2)-math.lgamma(nu/2)-0.5*math.log(nu*math.pi)
    return c-(nu+1)/2*torch.log1p(z*z/nu)
out={}
for k,v in sd.items():
    if v.ndim!=2: out[k]=v; continue
    X=v.double()
    r=torch.ones(X.shape[0],1,dtype=X.dtype); 
    for _ in range(3):
        c=(X/r).abs().median(0,keepdim=True).values/0.6745
        c=c.clamp_min(1e-8)
        r=(X/c).abs().median(1,keepdim=True).values/0.6745
        r=r.clamp_min(1e-3*r.median())
    S=r*c
    a=X.abs().flatten()
    T=torch.quantile(a[::max(1,a.numel()//2000000)],0.995).item()
    sn=T/NormalDist().inv_cdf(1-0.005/PI/2)
    z=X/S
    logf=tlogpdf(z,NU)-torch.log(S)
    logg=-0.5*(X/sn)**2-0.5*math.log(2*math.pi)-math.log(sn)
    p=torch.sigmoid(math.log(PI/(1-PI))+logg-logf)
    if MODE=="soft": Y=(1-p)*X
    else: Y=torch.where(p>0.5,torch.zeros_like(X),X)
    out[k]=Y.float()
    print(f"{k:26s} sn={sn:.3f} sn/std_clean={sn/Y.std().item():.2f} flagged={(p>0.5).double().mean().item():.4f} sum_p={p.mean().item():.4f}",flush=True)
save_weights(out,OUT)
