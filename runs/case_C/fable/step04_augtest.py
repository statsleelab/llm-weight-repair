"""Weight-space test: does augmenting with the paired matrix (same neuron / head-dim axis) improve
simulated-block completion?  Also check smaller ridge for wte."""
import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("cand_ridge.safetensors"); log=json.load(open("step03_log.json"))
LAMS=[1e-2,3e-2,1e-1,3e-1,1,3]
def sim(M,E,R,C,nrow,ncol,g,w=1.0,nsim=3):
    # M: samples x feats ; block of nrow samples x ncol feats; E extra feats (samples x e)
    keepr=torch.ones(M.shape[0],dtype=torch.bool); keepr[R[0]:R[1]+1]=False
    keepc=torch.ones(M.shape[1],dtype=torch.bool); keepc[C[0]:C[1]+1]=False
    Mi=M[keepr][:,keepc].double(); Ei=None if E is None else E[keepr].double()*w
    errs=np.zeros(len(LAMS))
    for s in range(nsim):
        r0=int(torch.randint(0,Mi.shape[0]-nrow,(1,),generator=g)); c0=int(torch.randint(0,Mi.shape[1]-ncol,(1,),generator=g))
        r=torch.zeros(Mi.shape[0],dtype=torch.bool); r[r0:r0+nrow]=True
        c=torch.zeros(Mi.shape[1],dtype=torch.bool); c[c0:c0+ncol]=True
        A=Mi[r][:,c]; X=Mi[:,~c]
        if Ei is not None: X=torch.cat([X,Ei],1)
        B=X[r]; D=X[~r]; Cm=Mi[~r][:,c]
        U,S,Vh=torch.linalg.svd(D,full_matrices=False); P=B@Vh.T; Q=U.T@Cm
        for i,lam in enumerate(LAMS):
            l=lam*(S**2).mean(); errs[i]+=(((P*(S/(S**2+l)))@Q-A)**2).sum().item()/(A**2).sum().item()/nsim
    return errs.round(3)
for L in [3,8]:
    fc=sd[f"h.{L}.mlp.c_fc.weight"]; pj=sd[f"h.{L}.mlp.c_proj.weight"]
    lf=log[f"h.{L}.mlp.c_fc.weight"]; lp=log[f"h.{L}.mlp.c_proj.weight"]
    for w in [None,0.5,1,2]:
        g=torch.Generator().manual_seed(1)
        print(L,"c_fc  aug w",w,sim(fc.T,None if w is None else pj,lf["cols"],lf["rows"],307,77,g,w or 1))
        g=torch.Generator().manual_seed(1)
        print(L,"c_proj aug w",w,sim(pj,None if w is None else fc.T,lp["rows"],lp["cols"],307,77,g,w or 1))
    ap=sd[f"h.{L}.attn.c_proj.weight"]; ca=sd[f"h.{L}.attn.c_attn.weight"]; la=log[f"h.{L}.attn.c_proj.weight"]
    for w in [None,0.5,1,2]:
        g=torch.Generator().manual_seed(1)
        print(L,"attn c_proj aug V w",w,sim(ap,None if w is None else ca[:,1536:].T,la["rows"],la["cols"],77,77,g,w or 1))
