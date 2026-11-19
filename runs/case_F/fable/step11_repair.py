# v4: local-fdr empirical Bayes.  Noise model (measured): ~3% of entries got additive
# N(0, sn^2), sn ~= 10 x matrix std.  For each matrix: baseline M (median polish [+ low rank]),
# residual R = X-M, local scale S = r_i c_j (robust), z = |R|/S.  In z-bins, the expected number of
# noise entries per row / column is computed analytically; the excess of observed counts over that
# is the genuine-heavy-tail occupancy, modelled rank-1 over (row, col).  p(corrupt) = noise/(noise+genuine).
# Output = X - p*R (posterior-mean shrinkage towards the baseline).
# (step11: significance-thresholded row/col boosts, A = number of Poisson sds)
# usage: python step11_repair.py RANK OUT [KEEP=4.5] [A=5]
import torch, sys, math
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights
from statistics import NormalDist
RANK=int(sys.argv[1]); OUT=sys.argv[2]
KEEP=float(sys.argv[3]) if len(sys.argv)>3 else 4.5
A=float(sys.argv[4]) if len(sys.argv)>4 else 2.5
PI=0.03
EDGES=[0,1,2,3,4,5,7,10,15,25,50,1e30]
sd = load_weights("damaged.safetensors")
def scales(R):
    r=torch.ones(R.shape[0],1)
    for _ in range(3):
        c=((R/r).abs().median(0,keepdim=True).values/0.6745); c=c.clamp_min(1e-3*c.median()+1e-12)
        r=((R/c).abs().median(1,keepdim=True).values/0.6745); r=r.clamp_min(1e-3*r.median()+1e-12)
    return r*c
def post(X,M,sn):
    R=X-M; S=scales(R); z=R.abs()/S
    sig=torch.sqrt(sn*sn+S*S)*math.sqrt(2)
    p=torch.zeros_like(R); N=R.numel(); nr,nc=R.shape
    for b in range(len(EDGES)-1):
        lo,hi=EDGES[b],EDGES[b+1]
        e=PI*(torch.erf(hi*S/sig)-torch.erf(lo*S/sig))      # per-entry prob: corrupted & in this bin
        inb=((z>=lo)&(z<hi)).float()
        Eg=e.sum().item(); Cg=inb.sum().item()
        gg=max(Cg-Eg,0.0)/N+1e-12
        Er=e.sum(1,keepdim=True); Ec=e.sum(0,keepdim=True)
        xr=((inb.sum(1,keepdim=True)-Er-A*torch.sqrt(Er+1)).clamp_min(0)/nc)/gg
        xc=((inb.sum(0,keepdim=True)-Ec-A*torch.sqrt(Ec+1)).clamp_min(0)/nr)/gg
        gen=gg*xr.clamp_min(1)*xc.clamp_min(1)
        pb=e/(e+gen+1e-30)
        p=p+inb*pb
    p=torch.where(R.abs()>KEEP*sn,torch.zeros_like(p),p)
    return p,R
def repair(X,sn):
    cm=X.median(0,keepdim=True).values
    rm=(X-cm).median(1,keepdim=True).values
    cm=cm+(X-cm-rm).median(0,keepdim=True).values
    M=cm+rm
    p,R=post(X,M,sn)
    for it in range(2 if RANK>0 else 0):
        Y=X-p*R-(cm+rm)
        U,s,V=torch.svd_lowrank(Y,q=RANK+8,niter=2)
        M=cm+rm+(U[:,:RANK]*s[:RANK])@V[:,:RANK].T
        p,R=post(X,M,sn)
    return X-p*R,p
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
