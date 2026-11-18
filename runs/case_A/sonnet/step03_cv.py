import torch,numpy as np,time
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
g=torch.Generator().manual_seed(0)
def impute(W,known,r,iters=30):
    X=torch.where(known,W,torch.zeros_like(W))
    # init: zeros -> iterate
    Z=X.clone()
    for it in range(iters):
        U,S,Vh=torch.linalg.svd(Z,full_matrices=False)
        L=(U[:,:r]*S[:r])@Vh[:r]
        Z=torch.where(known,W,L)
    return L
for name in ["h.0.attn.c_attn.weight","h.5.mlp.c_fc.weight","h.5.mlp.c_proj.weight","h.8.attn.c_proj.weight","wpe.weight"]:
    W=sd[name]; known=W!=0
    hold=(torch.rand(W.shape,generator=g)<0.02)&known
    k2=known&~hold
    tv=W[hold]; print(name,"rms true %.4f"%tv.pow(2).mean().sqrt())
    for r in [8,32,64,128,256]:
        if r>=min(W.shape): continue
        t=time.time();L=impute(W,k2,r,15)
        print("  r",r,"rmse %.4f"%(L[hold]-tv).pow(2).mean().sqrt(),"%.0fs"%(time.time()-t),flush=True)
