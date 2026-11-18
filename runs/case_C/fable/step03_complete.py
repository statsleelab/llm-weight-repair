"""Block completion: each 2-D matrix has one contiguous zeroed rectangle A (rows R x cols C).
Estimate A ~= B f(D) Cm  (ridge-regularised Nystrom / regression completion), where
M = [[A,B],[Cm,D]].  Ridge strength picked per matrix by simulating same-size blocks in the
intact part of the matrix (pure weight-space validation, no text involved)."""
import torch, numpy as np, json, time
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
LAMS=[1e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30]   # relative to mean s^2
def parts(M,R,C):
    r=torch.zeros(M.shape[0],dtype=torch.bool); r[R]=True
    c=torch.zeros(M.shape[1],dtype=torch.bool); c[C]=True
    return M[r][:,~c], M[~r][:,c], M[~r][:,~c]   # B, Cm, D
def factors(M,R,C):
    B,Cm,D=parts(M,R,C)
    U,S,Vh=torch.linalg.svd(D.double(),full_matrices=False)
    P=B.double()@Vh.T; Q=U.T@Cm.double()
    return P,S,Q
def est(P,S,Q,lam):
    l=lam*(S**2).mean()
    return (P*(S/(S**2+l)))@Q
out={}; log={}
g=torch.Generator().manual_seed(0)
for k in sorted(sd):
    M=sd[k]
    if M.ndim!=2: out[k]=M; continue
    t=time.time()
    z=(M==0)
    R=torch.nonzero(z.any(1))[:,0]; C=torch.nonzero(z.any(0))[:,0]
    assert int(z.sum())==len(R)*len(C) and z[R][:,C].all() and R[-1]-R[0]==len(R)-1 and C[-1]-C[0]==len(C)-1
    # simulation on intact sub-matrix
    keepr=torch.ones(M.shape[0],dtype=torch.bool); keepr[R]=False
    keepc=torch.ones(M.shape[1],dtype=torch.bool); keepc[C]=False
    Mi=M[keepr][:,keepc]
    errs=np.zeros(len(LAMS)); nsim=2 if M.shape[0]>10000 else 3
    for s in range(nsim):
        r0=int(torch.randint(0,Mi.shape[0]-len(R),(1,),generator=g)); c0=int(torch.randint(0,Mi.shape[1]-len(C),(1,),generator=g))
        Rs=torch.arange(r0,r0+len(R)); Cs=torch.arange(c0,c0+len(C))
        A=Mi[Rs][:,Cs].double()
        P,S,Q=factors(Mi,Rs,Cs)
        for i,lam in enumerate(LAMS):
            errs[i]+=((est(P,S,Q,lam)-A)**2).sum().item()/(A**2).sum().item()/nsim
    bi=int(errs.argmin()); lam=LAMS[bi]
    P,S,Q=factors(M,R,C)
    A=est(P,S,Q,lam).float()
    new=M.clone(); new[R[0]:R[-1]+1,C[0]:C[-1]+1]=A
    out[k]=new
    log[k]=dict(rows=[int(R[0]),int(R[-1])],cols=[int(C[0]),int(C[-1])],lam=lam,sim_rel_err=float(errs[bi]),errs=[round(float(e),4) for e in errs],
                fill_rms=float(A.pow(2).mean().sqrt()),mat_rms=float(Mi.pow(2).mean().sqrt()))
    print(f"{k:26s} R={log[k]['rows']} C={log[k]['cols']} lam={lam:g} sim_rel_err={errs[bi]:.3f} fill_rms={log[k]['fill_rms']:.4f} mat_rms={log[k]['mat_rms']:.4f} {time.time()-t:.1f}s",flush=True)
json.dump(log,open("step03_log.json","w"),indent=1)
save_weights(out,"cand_ridge.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
print("ppl damaged",perplexity(sd,ids,max_tokens=8192),"ppl ridge-completed",perplexity(out,ids,max_tokens=8192))
