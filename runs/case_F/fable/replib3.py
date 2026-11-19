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
RANK=0; OUT=None
KEEP=4.5
A=2.5
PI=0.03
FTH=0.7
ND=NormalDist()
AB=[0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,3.5,4,4.5,5,6,8,1e9]
EDGES=[0,1,2,3,4,5,6,8,10,13,16,20,25,32,40,50,64,80,100,130,160,200,1e30]
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
    # global absolute-magnitude override: where the matrix-wide count of entries of this magnitude is
    # explained by the noise model alone (fdr>=FTH), nothing of that size is genuine -> flag regardless of
    # local clustering.  Beyond 5 sn noise is impossible -> cap p by the global fdr.
    a=R.abs()/sn
    for b in range(len(AB)-1):
        lo,hi=AB[b],AB[b+1]
        inb=(a>=lo)&(a<hi)
        obs=inb.sum().item()
        if obs==0: continue
        ex=PI*N*2*(ND.cdf(hi)-ND.cdf(lo))
        fdr=min(1.0,ex/obs)
        if fdr>=FTH: p=torch.where(inb,torch.clamp(p,min=fdr),p)
        elif lo>=5: p=torch.where(inb,torch.clamp(p,max=fdr),p)
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

def repair_matrix(k,v):
    a=v.abs().flatten()
    T=torch.quantile(a[::max(1,a.numel()//2000000)].double(),0.995).item()
    sn=T/NormalDist().inv_cdf(1-0.005/PI/2)
    if "c_attn" in k:
        parts=[repair(b,sn) for b in v.split(768,dim=1)]
        return torch.cat([q[0] for q in parts],1).float()
    return repair(v,sn)[0].float()
