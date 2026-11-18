import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
from step11_null import nullmap, K
sd = load_weights("damaged.safetensors")
def maxrect(S):
    n,m=S.shape; best=(-1e9,None)
    cs=np.vstack([np.zeros(m),np.cumsum(S,0)])
    for a in range(n):
        for b in range(a+1,min(n,a+80)+1):
            col=cs[b]-cs[a]
            # kadane
            cur=0;start=0
            for j,x in enumerate(col):
                if cur<=0: cur=x;start=j
                else: cur+=x
                if cur>best[0]: best=(cur,(a,b,start,j+1))
    return best
def fine_stat(v,g,axis_idx,rows,cols):
    x=v[rows][:,cols]/g
    return x
out={}
for k,v in sd.items():
    if v.dim()!=2: continue
    s=nullmap(v).numpy()
    r=np.argsort(np.argsort(s.flatten())).reshape(s.shape)/s.size
    S=np.where(r<0.03,1.0,-0.08)
    if k=="wte.weight":
        # too tall for O(n^2 m); restrict to row-ranges via coarse pass
        rp=(r<0.03).mean(1); top=np.argsort(-np.convolve(rp,np.ones(20),'same'))[0]
        lo=max(0,top-80); S2=S[lo:top+80]; sc,(a,b,c,d)=maxrect(S2); a+=lo;b+=lo
    else:
        sc,(a,b,c,d)=maxrect(S)
    out[k]=[a*K,b*K,c*K,d*K]
    area=(b-a)*(d-c)*K*K/v.numel()
    print(f"{k:26s} score={sc:6.1f} rows={a*K}-{b*K} cols={c*K}-{d*K} area={area:.4f}",flush=True)
pass
