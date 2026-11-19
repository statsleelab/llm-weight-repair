import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
B=json.load(open("blocks8.json")); B5=json.load(open("blocks6.json")); out={}
def prof(W, a, h, c0, c1, s2):
    # per-row null-exact statistic using cols [c0,c1); rows far from [a,a+h) define structure
    n=W.shape[0]; far=torch.ones(n,dtype=torch.bool); far[max(0,a-h//2-4):a+h+h//2+4]=False
    X=W[:,c0:c1]; L=c1-c0
    A=X[far]
    e,V=torch.linalg.eigh(A.T@A)
    t=torch.zeros(n,dtype=torch.double)
    for k in (4,16,48):
        if k>=L//2: continue
        P=X@V[:,-k:]; f=P.pow(2).sum(1)/X.pow(2).sum(1).clamp_min(1e-12); t+=((f-k/L)/np.sqrt(2*k*(L-k)/(L*L*(L+2)))).clamp(-8,8)
    U=X.pow(2)/s2
    z=(U.mean(1)-1)/np.sqrt(2/L); k4=((U**2).mean(1)/3-1)/np.sqrt(96/9/L)
    t+=((z**2+k4.clamp(-6,6)**2).clamp(max=30)-2)/2
    return t.numpy(), far.numpy()
def best(t,h,a,rng):
    c=np.concatenate([[0],np.cumsum(t)]); n=len(t)
    lo=max(0,a-rng); hi=min(n-h,a+rng)
    cost=np.array([c[s+h]-c[s] for s in range(lo,hi+1)]); s=int(cost.argmin())
    at=lambda d: cost[s+d]-cost[s] if 0<=s+d<len(cost) else float('nan')
    return lo+s,[at(-8),at(-2),at(2),at(8)]
for k,b in B.items():
    W=sd[k].double(); s2=W.var().item()
    r0,r1,c0,c1=b; h=r1-r0; w=c1-c0
    for it in range(4):
        t,far=prof(W,r0,h,c0+w//10,c1-w//10,s2)
        nr0,mr=best(t,h,r0,h//2); tin=t[nr0:nr0+h].mean(); tout=t[far].mean()
        t2,far2=prof(W.T,c0,w,nr0+h//10,nr0+h-h//10,s2)
        nc0,mc=best(t2,w,c0,w//2); tin2=t2[nc0:nc0+w].mean(); tout2=t2[far2].mean()
        ch=(nr0!=r0) or (nc0!=c0); r0,c0=nr0,nc0; r1,c1=r0+h,c0+w
        if not ch: break
    X=W[r0:r1,c0:c1]
    flag="" if (r0==b[0] and c0==b[2]) else "  <-- moved"
    print(f"{k:26s} was ({b[0]},{b[2]}) now ({r0},{c0}) it={it} rows in/out={tin:.2f}/{tout:.1f} cols in/out={tin2:.2f}/{tout2:.1f} rowmarg={np.round(mr,0)} colmarg={np.round(mc,0)} Xstd/σ={X.std()/W.std():.3f}{flag}",flush=True)
    out[k]=[r0,r1,c0,c1]
json.dump(out,open("blocks9.json","w"))
