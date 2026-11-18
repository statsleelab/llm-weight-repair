"""Ridge-regression imputation of zeroed contiguous blocks; method/lambda chosen per matrix
by cross-validation on simulated blocks inside the intact part of the same matrix."""
import torch, json, math
torch.set_num_threads(1)
torch.manual_seed(0)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")

def find_block(v):
    z=(v==0); r=z.sum(1); c=z.sum(0)
    R=(r>=r.max()*0.5).nonzero()[:,0]; C=(c>=c.max()*0.5).nonzero()[:,0]
    return R.min().item(), R.max().item()+1, C.min().item(), C.max().item()+1

LAMS=[1e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100]
def ridge_preds(F, Y, Fnew, max_samples=12000):
    """F: samples x p, Y: samples x t, Fnew: q x p. return dict lam-> q x t prediction (lam relative to mean s^2)"""
    F=F.double(); Y=Y.double(); Fnew=Fnew.double()
    if F.shape[0]>max_samples:
        idx=torch.randperm(F.shape[0])[:max_samples]; F=F[idx]; Y=Y[idx]
    mu_f=F.mean(0); mu_y=Y.mean(0)
    F=F-mu_f; Fn=Fnew-mu_f; Yc=Y-mu_y
    U,S,Vh=torch.linalg.svd(F,full_matrices=False)
    UtY=U.T@Yc; FnV=Fn@Vh.T
    s2m=(S**2).mean()
    out={}
    for lam in LAMS:
        d=S/(S**2+lam*s2m*1e-2)
        out[lam]=(FnV*d)@UtY+mu_y
    return out

def impute(W, r0,r1,c0,c1, orient):
    """returns dict lam -> predicted block. W has the block region considered unknown."""
    n,m=W.shape
    rm=torch.ones(n,dtype=bool); rm[r0:r1]=False
    cm=torch.ones(m,dtype=bool); cm[c0:c1]=False
    if orient=="col":   # samples = known rows; features = other cols; targets = block cols
        return ridge_preds(W[rm][:,cm], W[rm][:,c0:c1], W[r0:r1][:,cm])
    else:               # samples = known cols; features = other rows; targets = block rows
        P=ridge_preds(W[:,cm][rm].T, W[:,cm][r0:r1].T, W[rm][:,c0:c1].T)
        return {k:v.T for k,v in P.items()}

def cv(W, r0,r1,c0,c1, orients, ntr=3):
    n,m=W.shape; h=r1-r0; w=c1-c0
    rm=torch.ones(n,dtype=bool); rm[r0:r1]=False
    cm=torch.ones(m,dtype=bool); cm[c0:c1]=False
    K=W[rm][:,cm]  # intact submatrix
    nn,mm=K.shape
    errs={}
    for t in range(ntr):
        a=torch.randint(0,nn-h+1,(1,)).item(); b=torch.randint(0,mm-w+1,(1,)).item()
        truth=K[a:a+h,b:b+w].double()
        for o in orients:
            for lam,pred in impute(K,a,a+h,b,b+w,o).items():
                errs.setdefault((o,lam),[]).append(((pred-truth).norm()/truth.norm()).item())
        # averaged orientation
        if len(orients)==2:
            P1=impute(K,a,a+h,b,b+w,orients[0]); P2=impute(K,a,a+h,b,b+w,orients[1])
            for l1 in LAMS:
                for l2 in LAMS:
                    pred=(P1[l1]+P2[l2])/2
                    errs.setdefault(("avg",(l1,l2)),[]).append(((pred-truth).norm()/truth.norm()).item())
    return {k:sum(v)/len(v) for k,v in errs.items()}

out=dict(sd); log={}
for k,v in sd.items():
    if v.dim()!=2: continue
    r0,r1,c0,c1=find_block(v)
    orients=["col"] if k=="wte.weight" else ["col","row"]
    E=cv(v,r0,r1,c0,c1,orients, ntr=2 if k=="wte.weight" else 3)
    best=min(E,key=E.get)
    o,lam=best
    if o=="avg":
        P1=impute(v,r0,r1,c0,c1,"col")[lam[0]]; P2=impute(v,r0,r1,c0,c1,"row")[lam[1]]; pred=(P1+P2)/2
    else:
        pred=impute(v,r0,r1,c0,c1,o)[lam]
    w=v.clone(); w[r0:r1,c0:c1]=pred.float(); out[k]=w
    bc=min((e,kk) for kk,e in E.items() if kk[0]=="col"); br=min([(e,kk) for kk,e in E.items() if kk[0]=="row"] or [(9,None)])
    log[k]=dict(block=[r0,r1,c0,c1],best=str(best),cv_relerr=E[best],best_col=bc[0],best_row=br[0])
    print(f"{k:26s} block[{r0}:{r1},{c0}:{c1}] best={best} cv_relerr={E[best]:.3f} (col {bc[0]:.3f}, row {br[0]:.3f}) pred_norm/rowstd={pred.std().item()/v.std().item():.2f}",flush=True)
save_weights(out,"cand_ridge.safetensors")
json.dump(log,open("step05_log.json","w"),indent=1)
