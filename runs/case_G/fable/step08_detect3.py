import torch, numpy as np, json, time
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
out={}
def score(W):
    # cross-validated row-gram score, entrywise
    G=W@W.T; G.fill_diagonal_(0)
    c=W.pow(2).sum(0,keepdim=True)
    S=W*(G@W)-W.pow(2)*(c-W.pow(2))
    return S
for k,W0 in sd.items():
    if W0.ndim!=2: continue
    t=time.time()
    W=W0.double(); n,m=W.shape
    for _ in range(2):
        W=W/W.pow(2).mean(1,keepdim=True).sqrt().clamp_min(1e-8)
        W=W/W.pow(2).mean(0,keepdim=True).sqrt().clamp_min(1e-8)
    S=score(W.T).T
    S=S/S.std()
    if n<4000:
        S2=score(W); S=S+S2/S2.std()
    S=S-S.median()
    S=S.clamp(-5*S.std(),5*S.std())
    h=round(.1732*n); w=round(.1732*m)
    I=torch.zeros(n+1,m+1,dtype=torch.double); I[1:,1:]=S.cumsum(0).cumsum(1)
    R=I[h:,w:]-I[:-h,w:]-I[h:,:-w]+I[:-h,:-w]
    idx=R.argmin().item(); r0,c0=divmod(idx,R.shape[1]); best=R[r0,c0].item()
    R2=R.clone(); R2[max(0,r0-h):r0+h, max(0,c0-w):c0+w]=1e18; second=R2.min().item()
    bg=R2[::h//2,::w//2].flatten(); bg=bg[bg<1e17]
    print(f"{k:26s} rows[{r0},{r0+h}) cols[{c0},{c0+w}) best={best/(h*w):.3f} second={second/(h*w):.3f} bgmed={bg.median().item()/(h*w):.3f} bgstd={bg.std().item()/(h*w):.3f} [{time.time()-t:.1f}s]",flush=True)
    out[k]=[r0,r0+h,c0,c0+w]
json.dump(out,open("blocks3.json","w"))
