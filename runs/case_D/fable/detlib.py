import torch, numpy as np
def boxsum(M,h,w):
    I=np.zeros((M.shape[0]+1,M.shape[1]+1)); I[1:,1:]=M.cumsum(0).cumsum(1)
    return I[h:,w:]-I[:-h,w:]-I[h:,:-w]+I[:-h,:-w]
def norm1(S):
    r=S.mean(1,keepdims=True); c=S.mean(0,keepdims=True)
    return S/(r*c/S.mean())
def loo_resid(W):
    W=W.double(); n,m=W.shape
    if n<=m:
        G=W@W.T/m; lam=0.02*G.diagonal().mean()
        Th=torch.linalg.inv(G+lam*torch.eye(n,dtype=W.dtype)); E=(Th@W)/Th.diagonal()[:,None]
    else:
        G=W.T@W/n; lam=0.02*G.diagonal().mean()
        Th=torch.linalg.inv(G+lam*torch.eye(m,dtype=W.dtype)); E=(W@Th)/Th.diagonal()[None,:]
    return E
def best_interval(s, thr):
    # max-sum contiguous interval of (s-thr)
    d=s-thr; c=np.concatenate([[0],d.cumsum()])
    best=(-1e18,0,0); mn=0; mi=0
    for e in range(1,len(c)):
        if c[e-1]<mn: mn=c[e-1]; mi=e-1
        if c[e]-mn>best[0]: best=(c[e]-mn,mi,e)
    return best[1],best[2]
def refine(S,r0,r1,c0,c1,iters=4):
    n,m=S.shape
    for _ in range(iters):
        s=S[:,c0:c1].mean(1); ins=s[r0:r1].mean(); out=np.median(np.delete(s,np.s_[r0:r1]))
        r0,r1=best_interval(s,(ins+out)/2)
        s=S[r0:r1].mean(0); ins=s[c0:c1].mean(); out=np.median(np.delete(s,np.s_[c0:c1]))
        c0,c1=best_interval(s,(ins+out)/2)
    s=S[:,c0:c1].mean(1); rin=s[r0:r1].mean(); rout=np.delete(s,np.s_[r0:r1])
    return r0,r1,c0,c1,rin,np.median(rout),rout.std()
