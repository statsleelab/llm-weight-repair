import torch, numpy as np, json, time
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
B=json.load(open("blocks_final.json"))
LAMS=[1e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1,3,10]
def solve(A,Bm,D,lams):
    # est = D (A^T A + lam)^-1 A^T B   for each lam (lam relative to top eigenvalue)
    if A.shape[0]>=A.shape[1]:
        e,V=torch.linalg.eigh(A.T@A); AtB=V.T@(A.T@Bm); DV=D@V
        return [DV@(AtB/(e.clamp_min(0)+l*e[-1])[:,None]) for l in lams]
    else:
        e,U=torch.linalg.eigh(A@A.T); UB=U.T@Bm; DAU=D@(A.T@U)
        return [DAU@(UB/(e.clamp_min(0)+l*e[-1])[:,None]) for l in lams]
rep={k:v.clone() for k,v in sd.items()}; info={}
for k,(r0,r1,c0,c1) in B.items():
    t=time.time(); W=sd[k].double(); n,m=W.shape; h=r1-r0; w=c1-c0
    rm=torch.ones(n,dtype=torch.bool); rm[r0:r1]=False
    cm=torch.ones(m,dtype=torch.bool); cm[c0:c1]=False
    # fake block disjoint in rows and cols
    fr=(r0+n//2)%n; fr=min(fr,n-h); fc=(c0+m//2)%m; fc=min(fc,m-w)
    rm2=rm.clone(); rm2[fr:fr+h]=False; cm2=cm.clone(); cm2[fc:fc+w]=False
    Wr=W[rm2]
    ests=solve(Wr[:,cm2],Wr[:,fc:fc+w],W[fr:fr+h][:,cm2],LAMS)
    T=W[fr:fr+h,fc:fc+w]
    errs=[((e-T).pow(2).sum()/T.pow(2).sum()).item() for e in ests]
    j=int(np.argmin(errs))
    Wc=W[rm]
    est=solve(Wc[:,cm],Wc[:,c0:c1],W[r0:r1][:,cm],[LAMS[j]])[0]
    rep[k][r0:r1,c0:c1]=est.float()
    info[k]=dict(lam=LAMS[j],val_relerr=errs[j],est_std=est.std().item(),sigma=W.std().item())
    print(f"{k:26s} lam={LAMS[j]:<7g} val rel sq err={errs[j]:.3f} est std={est.std():.4f} σ={W.std():.4f} [{time.time()-t:.1f}s]",flush=True)
json.dump(info,open("repair_info.json","w"))
save_weights(rep,"repaired.safetensors")
