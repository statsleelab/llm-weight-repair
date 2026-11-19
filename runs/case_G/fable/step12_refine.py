import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
B=json.load(open("blocks5.json")); out={}
def best(q,h,a,rng):
    c=np.concatenate([[0],np.cumsum(q)]); n=len(q)
    lo=max(0,a-rng); hi=min(n-h,a+rng)
    cost=np.array([c[s+h]-c[s] for s in range(lo,hi+1)])
    s=int(cost.argmin())
    def at(d):
        j=s+d
        return cost[j]-cost[s] if 0<=j<len(cost) else float('nan')
    return lo+s,(at(-10),at(-3),at(3),at(10))
for k,b in B.items():
    W=sd[k].double(); n,m=W.shape; U=W.pow(2)/W.var()
    r0,r1,c0,c1=b; h=r1-r0; w=c1-c0
    if k=="wte.weight": rng_r,rng_c=600,20
    else: rng_r,rng_c=h//2,w//2
    for it in range(4):
        ci0,ci1=c0+w//10,c1-w//10
        z=(U[:,ci0:ci1].mean(1)-1)/np.sqrt(2/(ci1-ci0)); k4=((U[:,ci0:ci1]**2).mean(1)/3-1)/np.sqrt(96/9/(ci1-ci0))
        q=(z**2+k4.clamp(-6,6)**2).clamp(max=30).numpy()
        nr0,mr=best(q,h,r0,rng_r)
        ri0,ri1=nr0+h//10,nr0+h-h//10
        z=(U[ri0:ri1].mean(0)-1)/np.sqrt(2/(ri1-ri0)); k4=((U[ri0:ri1]**2).mean(0)/3-1)/np.sqrt(96/9/(ri1-ri0))
        q2=(z**2+k4.clamp(-6,6)**2).clamp(max=30).numpy()
        nc0,mc=best(q2,w,c0,rng_c)
        ch=(nr0!=r0) or (nc0!=c0); r0,c0=nr0,nc0; r1,c1=r0+h,c0+w
        if not ch: break
    X=W[r0:r1,c0:c1]
    print(f"{k:26s} was ({b[0]},{b[2]}) now rows[{r0},{r1}) cols[{c0},{c1}) it={it} rowmargins={np.round(mr,0)} colmargins={np.round(mc,0)} Xstd/σ={X.std()/W.std():.3f} kurt={(X**4).mean()/X.var()**2:.2f}",flush=True)
    out[k]=[r0,r1,c0,c1]
json.dump(out,open("blocks6.json","w"))
