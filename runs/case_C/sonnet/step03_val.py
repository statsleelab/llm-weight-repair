import torch,numpy as np,sys
torch.set_num_threads(1)
from gpt2_lib import *
from imp import impute
sd=load_weights("damaged.safetensors")
rng=np.random.default_rng(0)
def test(name,W,axis,lams):
    W=W.numpy().astype(np.float64)
    Z=(W==0)
    X=W if axis==0 else W.T
    Zx=Z if axis==0 else Z.T
    hold=(rng.random(X.shape)<0.01)&~Zx
    Xh=X.copy(); Xh[hold]=0
    M=Zx|hold
    base=np.sqrt((X[hold]**2).mean())
    out=[]
    for lam in lams:
        Xi=impute(Xh,M,lam=lam,iters=2)
        e=np.sqrt(((Xi[hold]-X[hold])**2).mean())
        out.append(f"lam{lam}:{e/base:.3f}")
    print(name,axis,out,flush=True)
test("wpe.weight",sd["wpe.weight"],0,[0.02,0.1,0.3,0.6])
test("h.5.attn.c_attn.weight",sd["h.5.attn.c_attn.weight"],1,[0.1,0.3,0.6])
test("h.5.attn.c_proj.weight",sd["h.5.attn.c_proj.weight"],0,[0.1,0.3,0.6])
test("h.5.mlp.c_fc.weight",sd["h.5.mlp.c_fc.weight"],1,[0.6,0.9])
# wpe smoothness: neighbor interpolation
W=sd["wpe.weight"].numpy().astype(np.float64); Z=W==0
hold=(rng.random(W.shape)<0.02)&~Z&(np.arange(1024)[:,None]>0)&(np.arange(1024)[:,None]<1023)
pred=0.5*(np.roll(W,1,0)+np.roll(W,-1,0))
ok=hold&~np.roll(Z,1,0)&~np.roll(Z,-1,0)
print("wpe neighbor interp ratio",np.sqrt(((pred-W)[ok]**2).mean()/(W[ok]**2).mean()))
