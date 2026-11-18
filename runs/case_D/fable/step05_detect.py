import torch, numpy as np, json, time
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
def boxsum(M,h,w):
    I=np.zeros((M.shape[0]+1,M.shape[1]+1)); I[1:,1:]=M.cumsum(0).cumsum(1)
    return I[h:,w:]-I[:-h,w:]-I[h:,:-w]+I[:-h,:-w]
def norm1(S):
    r=S.mean(1,keepdims=True); c=S.mean(0,keepdims=True)
    return S/(r*c/S.mean())
def loo_resid(W):
    W=W.double()
    n,m=W.shape
    if n<=m:   # rows as variables
        G=W@W.T/m; lam=0.02*G.diagonal().mean()
        Th=torch.linalg.inv(G+lam*torch.eye(n,dtype=W.dtype)); E=(Th@W)/Th.diagonal()[:,None]
    else:
        G=W.T@W/n; lam=0.02*G.diagonal().mean()
        Th=torch.linalg.inv(G+lam*torch.eye(m,dtype=W.dtype)); E=(W@Th)/Th.diagonal()[None,:]
    return E
if __name__!="__main__": raise SystemExit
res={}
for k,v in sd.items():
    if v.ndim!=2: continue
    t=time.time()
    n,m=v.shape; h,w=round(.1*n),round(.1*m)
    E=loo_resid(v).numpy(); W=v.double().numpy()
    out=[]
    for name,S in [("W2",norm1(W**2)),("E2",norm1(E**2)),("ratio",norm1(E**2/(W**2+1e-3*W.var())) if False else norm1(np.abs(E))/norm1(np.abs(W)+1e-9) if False else norm1(E**2)-norm1(W**2))]:
        F=boxsum(S,h,w)/(h*w)
        med=np.median(F); sc=np.median(np.abs(F-med))*1.4826
        i=np.unravel_index(F.argmax(),F.shape); j=np.unravel_index(F.argmin(),F.shape)
        out.append((name,i,(F[i]-med)/sc,j,(F[j]-med)/sc))
    res[k]=out
    print(f"{k:26s} {n}x{m} box {h}x{w} "+" | ".join(f"{a}: max@{tuple(map(int,b))} z={c:.1f} min@{tuple(map(int,d))} z={e:.1f}" for a,b,c,d,e in out), f"[{time.time()-t:.0f}s]",flush=True)
