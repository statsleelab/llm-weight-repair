import numpy as np, torch
def impute(w,a,b,l,e,k,iters=15):
    W=torch.tensor(w,dtype=torch.float32).clone()
    W[a:b,l:e]=0
    if k==0: return W.numpy()
    for _ in range(iters):
        U,S,V=torch.svd_lowrank(W,q=k,niter=2)
        approx=(U[a:b]*S)@V[l:e].T
        W[a:b,l:e]=approx
    return W.numpy()
