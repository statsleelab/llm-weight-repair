import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
B=json.load(open("blocks5.json")); out={}
def score(W):
    G=W@W.T; G.fill_diagonal_(0)
    c=W.pow(2).sum(0,keepdim=True)
    return W*(G@W)-W.pow(2)*(c-W.pow(2))
def qprof(Us):   # Us: items x window ; returns per-item q
    L=Us.shape[1]
    z=(Us.mean(1)-1)/np.sqrt(2/L); k4=((Us**2).mean(1)/3-1)/np.sqrt(96/9/L)
    return (z**2+k4.clamp(-6,6)**2).clamp(max=30).numpy()
def combine(profs,a,h):
    n=len(profs[0]); ins=np.zeros(n,bool); ins[a+h//8:a+h-h//8]=True
    outm=np.ones(n,bool); outm[max(0,a-h//2):a+h+h//2]=False
    tot=np.zeros(n); ds=[]
    for p in profs:
        mi,mo=p[ins].mean(),p[outm].mean(); v=(p[outm].var()+p[ins].var())/2+1e-12
        tot+= (mo-mi)/v*(p-(mi+mo)/2); ds.append((mo-mi)/np.sqrt(v))
    return tot,ds      # tot low (negative) inside
def best(t,h,a,rng):
    c=np.concatenate([[0],np.cumsum(t)]); n=len(t)
    lo=max(0,a-rng); hi=min(n-h,a+rng)
    cost=np.array([c[s+h]-c[s] for s in range(lo,hi+1)]); s=int(cost.argmin())
    at=lambda d: cost[s+d]-cost[s] if 0<=s+d<len(cost) else float('nan')
    return lo+s,[at(-8),at(-2),at(2),at(8)]
for k,b in B.items():
    W=sd[k].double(); n,m=W.shape; U=W.pow(2)/W.var()
    Wn=W.clone()
    for _ in range(2):
        Wn=Wn/Wn.pow(2).mean(1,keepdim=True).sqrt().clamp_min(1e-8)
        Wn=Wn/Wn.pow(2).mean(0,keepdim=True).sqrt().clamp_min(1e-8)
    S=score(Wn.T).T; S=S/S.std()
    if n<4000: S2=score(Wn); S=S+S2/S2.std()
    S=S.clamp(-6*S.std(),6*S.std())
    r0,r1,c0,c1=b; h=r1-r0; w=c1-c0
    for it in range(5):
        ci0,ci1=c0+w//10,c1-w//10
        t,dr=combine([qprof(U[:,ci0:ci1]),S[:,ci0:ci1].mean(1).numpy()],r0,h)
        nr0,mr=best(t,h,r0,h//2)
        ri0,ri1=nr0+h//10,nr0+h-h//10
        t,dc=combine([qprof(U[ri0:ri1].T),S[ri0:ri1].mean(0).numpy()],c0,w)
        nc0,mc=best(t,w,c0,w//2)
        ch=(nr0!=r0) or (nc0!=c0); r0,c0=nr0,nc0; r1,c1=r0+h,c0+w
        if not ch: break
    X=W[r0:r1,c0:c1]
    print(f"{k:26s} was ({b[0]},{b[2]}) now rows[{r0},{r1}) cols[{c0},{c1}) it={it} d'rows(q,S)={np.round(dr,1)} d'cols={np.round(dc,1)} rowmarg={np.round(mr,0)} colmarg={np.round(mc,0)} Xstd/σ={X.std()/W.std():.3f}",flush=True)
    out[k]=[r0,r1,c0,c1]
json.dump(out,open("blocks7.json","w"))
