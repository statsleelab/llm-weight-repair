import torch, math, time; torch.set_num_threads(1)
from gpt2_lib import *
torch.manual_seed(0)
sd = load_weights("damaged.safetensors")
def hard_impute(W, miss, k, iters=15):
    X=W.clone(); X[miss]=0
    for _ in range(iters):
        U,S,V=torch.svd_lowrank(X,q=k+10,niter=2)
        R=(U[:,:k]*S[:k])@V[:,:k].T
        X[miss]=R[miss]
    return X
def cv(name, ks):
    W=sd[name]; miss=(W==0)
    hold=(torch.rand(W.shape)<0.01)&~miss
    M=miss|hold
    t=W[hold]
    out=[]
    for k in ks:
        t0=time.time(); X=hard_impute(W,M,k)
        p=X[hold]
        r2=1-((p-t)**2).sum()/(t**2).sum()
        out.append((k,round(r2.item(),3),round(time.time()-t0,1)))
    print(name, out, flush=True)
for n in ["h.0.attn.c_attn.weight","h.5.attn.c_attn.weight","h.0.mlp.c_fc.weight","h.6.mlp.c_proj.weight","h.0.attn.c_proj.weight","h.11.mlp.c_proj.weight","wpe.weight"]:
    cv(n,[16,64,128,256])
