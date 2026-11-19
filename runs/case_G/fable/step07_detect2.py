import torch, numpy as np, json, time
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
out={}
def edges(p, lo, hi, a, b):
    # p: 1-D profile, low inside block; rough block [a,b). refine each edge by best step fit near it
    n=len(p); res=[]
    for e,sign in ((a,1),(b,-1)):
        best=None
        L=max(8,(b-a)//3)
        for c in range(max(1,e-L//2),min(n-1,e+L//2)+1):
            l=p[max(0,c-L):c]; r=p[c:min(n,c+L)]
            if len(l)<3 or len(r)<3: continue
            sc=sign*(l.mean()-r.mean())*np.sqrt(len(l)*len(r)/(len(l)+len(r)))
            if best is None or sc>best[0]: best=(sc,c)
        res.append(best[1])
    return res
for k,W0 in sd.items():
    if W0.ndim!=2: continue
    t=time.time()
    W=W0.double(); flip = W.shape[0]!=768
    if flip: W=W.T          # items = rows (768), long dim = columns
    n,m=W.shape
    W=W/W.pow(2).mean(0,keepdim=True).sqrt().clamp_min(1e-8)  # equalise long-dim scale
    nw=96; edges_w=np.linspace(0,m,nw+1).round().astype(int)
    G=W@W.T
    T=torch.zeros(n,nw,dtype=torch.double)
    for a in range(nw):
        Ww=W[:,edges_w[a]:edges_w[a+1]]; Gw=Ww@Ww.T; Gr=G-Gw
        Gw.fill_diagonal_(0); Gr.fill_diagonal_(0)
        T[:,a]=(Gw*Gr).sum(1)/((Gw.pow(2).sum(1)*Gr.pow(2).sum(1)).sqrt()+1e-12)
    med=T.median(1,keepdim=True).values; mad=(T-med).abs().median(1,keepdim=True).values*1.4826+1e-6
    Z=((T-med)/mad).clamp(-6,6)
    h=133; w=round(.1732*nw)
    I=torch.zeros(n+1,nw+1,dtype=torch.double); I[1:,1:]=Z.cumsum(0).cumsum(1)
    R=I[h:,w:]-I[:-h,w:]-I[h:,:-w]+I[:-h,:-w]
    idx=R.argmin().item(); r0,a0=divmod(idx,R.shape[1])
    best=R[r0,a0].item()/(h*w)
    # second best non-overlapping
    R2=R.clone(); R2[max(0,r0-h):r0+h, max(0,a0-w):a0+w]=1e9; second=R2.min().item()/(h*w)
    # refine items edges
    rowp=Z[:,a0:a0+w].mean(1).numpy()
    ra,rb=edges(rowp,0,n,r0,r0+h)
    # refine long edges with per-index score
    c0=edges_w[a0]; c1=edges_w[a0+w]; mar=(c1-c0)//4
    keep=torch.ones(m,dtype=torch.bool); keep[max(0,c0-mar):min(m,c1+mar)]=False
    Gr=W[:,keep]@W[:,keep].T
    Rm=torch.zeros(n,dtype=torch.bool); Rm[ra:rb]=True
    Gx=Gr[Rm][:,~Rm]              # |R| x rest
    colp=((W[Rm].T@Gx)*W[~Rm].T).sum(1).numpy()   # per long index
    colp=colp/np.median(np.abs(colp))
    ca,cb=edges(colp,0,m,c0,c1)
    name=("cols","rows") if not flip else ("rows","cols")
    print(f"{k:26s} zin={best:.2f} z2={second:.2f} items[{ra},{rb}) n={rb-ra} long[{ca},{cb}) n={cb-ca} exp={round(.1732*m)} in/out colp {colp[ca:cb].mean():.2f}/{np.median(colp):.2f} [{time.time()-t:.1f}s]",flush=True)
    out[k]=[ra,rb,int(ca),int(cb)] if not flip else [int(ca),int(cb),ra,rb]
json.dump(out,open("blocks.json","w"))
