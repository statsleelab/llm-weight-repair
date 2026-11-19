import torch, numpy as np, json, time
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
out={}
K=48
for k,W in sd.items():
    if W.ndim!=2: continue
    t=time.time()
    W=W.double(); n,m=W.shape
    if m<=n:
        e,V=torch.linalg.eigh(W.T@W); V=V[:,-K:]; L=(W@V)@V.T
    else:
        e,U=torch.linalg.eigh(W@W.T); U=U[:,-K:]; L=U@(U.T@W)
    # normalised score: explained fraction per entry relative to row/col scale
    S=W*L
    S=S/ (W.pow(2).mean(1,keepdim=True).sqrt()*W.pow(2).mean(0,keepdim=True).sqrt())
    S=S-S.mean()
    h=round(0.1732*n); w=round(0.1732*m)
    I=torch.zeros(n+1,m+1,dtype=torch.double); I[1:,1:]=S.cumsum(0).cumsum(1)
    R=I[h:,w:]-I[:-h,w:]-I[h:,:-w]+I[:-h,:-w]
    idx=R.argmin().item(); r0,c0=divmod(idx,R.shape[1])
    best=R[r0,c0].item()
    # background distribution of rectangle sums (subsample far away)
    bg=R[::max(1,h//2),::max(1,w//2)].flatten()
    z=(best-bg.median().item())/(1.4826*(bg-bg.median()).abs().median().item())
    # edge refinement profiles
    rowp=S[:,c0:c0+w].mean(1); colp=S[r0:r0+h].mean(0)
    thr_r=(rowp[r0+5:r0+h-5].mean()+rowp.median())/2; thr_c=(colp[c0+5:c0+w-5].mean()+colp.median())/2
    def sm(p,q=9):
        return torch.nn.functional.avg_pool1d(p[None,None],q,1,q//2,count_include_pad=False)[0,0]
    rb=(sm(rowp)<thr_r).nonzero().flatten(); cb=(sm(colp)<thr_c).nonzero().flatten()
    rb=rb[(rb>=r0-20)&(rb<r0+h+20)]; cb=cb[(cb>=c0-20)&(cb<c0+w+20)]
    print(f"{k:26s} {n}x{m} rect r0={r0} h={h} c0={c0} w={w} z={z:.1f} refined rows {rb.min().item()}..{rb.max().item()} ({len(rb)}) cols {cb.min().item()}..{cb.max().item()} ({len(cb)}) in={S[r0:r0+h,c0:c0+w].mean():.3f} [{time.time()-t:.1f}s]",flush=True)
    out[k]=[r0,r0+h,c0,c0+w]
json.dump(out,open("blocks.json","w"))
