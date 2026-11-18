import torch, numpy as np, json, time
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
blocks=json.load(open("blocks_final.json"))
LAMS=[0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0]
def ridge_preds(X,Y,Xn,lams):
    """X (s,f) Y (s,t) Xn (q,f) -> list of preds (q,t). lam scaled by mean diag."""
    s,f=X.shape
    if f<=s:
        G=X.T@X; XY=X.T@Y; sc=G.diagonal().mean(); I=torch.eye(f,dtype=X.dtype)
        return [Xn@torch.linalg.solve(G+l*sc*I,XY) for l in lams]
    K=X@X.T; sc=K.diagonal().mean()*s/f; I=torch.eye(s,dtype=X.dtype); KN=Xn@X.T
    return [KN@torch.linalg.solve(K+l*sc*I,Y) for l in lams]
def impute(W,R,C,keep_r,keep_c,lams):
    """predict W[R][:,C]; training uses rows keep_r (not in R) and cols keep_c (not in C)."""
    # B: columns as features, train on clean rows
    pB=ridge_preds(W[keep_r][:,keep_c],W[keep_r][:,C],W[R][:,keep_c],lams)
    # A: rows as features, train on clean cols
    pA=ridge_preds(W[keep_r][:,keep_c].T,W[R][:,keep_c].T,W[keep_r][:,C].T,lams)
    return [p.T for p in pA],pB
out_imp={k:v.clone() for k,v in sd.items()}; out_zero={k:v.clone() for k,v in sd.items()}
log={}
for k,(r0,r1,c0,c1) in blocks.items():
    t=time.time()
    W=sd[k].double(); n,m=W.shape; h,w=r1-r0,c1-c0
    rows=torch.arange(n); cols=torch.arange(m)
    R=rows[r0:r1]; C=cols[c0:c1]
    # pseudo block far from real block
    p0=(r0+n//2)%n; p0=min(p0,n-h); q0=(c0+m//2)%m; q0=min(q0,m-w)
    Rp=rows[p0:p0+h]; Cp=cols[q0:q0+w]
    mr=torch.ones(n,dtype=torch.bool); mr[R]=False; mc=torch.ones(m,dtype=torch.bool); mc[C]=False
    mr2=mr.clone(); mr2[Rp]=False; mc2=mc.clone(); mc2[Cp]=False
    pA,pB=impute(W,Rp,Cp,rows[mr2],cols[mc2],LAMS)
    T=W[Rp][:,Cp]; base=(T**2).mean()
    eA=[float(((p-T)**2).mean()/base) for p in pA]; eB=[float(((p-T)**2).mean()/base) for p in pB]
    ia=int(np.argmin(eA)); ib=int(np.argmin(eB))
    useA = eA[ia]<=eB[ib]
    coef=torch.tensor([1.0,0.0] if useA else [0.0,1.0],dtype=torch.float64)
    if min(eA[ia],eB[ib])>=1.0: coef=coef*0
    ecomb=min(eA[ia],eB[ib],1.0)
    # real
    pA,pB=impute(W,R,C,rows[mr],cols[mc],[LAMS[ia]]+[LAMS[ib]])
    est=coef[0]*pA[0]+coef[1]*pB[1]
    out_imp[k][r0:r1,c0:c1]=est.float(); out_zero[k][r0:r1,c0:c1]=0
    log[k]=dict(block=[r0,r1,c0,c1],relmse_A=eA[ia],lamA=LAMS[ia],relmse_B=eB[ib],lamB=LAMS[ib],coef=[float(c) for c in coef],relmse_comb=ecomb,
                est_std=float(est.std()),noise_std=float(W[R][:,C].std()),clean_std=float(W[mr][:,mc].std()))
    print(f"{k:26s} blk {r0},{c0} relMSE A {eA[ia]:.3f}(lam {LAMS[ia]}) B {eB[ib]:.3f}(lam {LAMS[ib]}) comb {ecomb:.3f} coef {coef.numpy().round(2)} est_std {est.std():.3f} blk_std {W[R][:,C].std():.3f} [{time.time()-t:.0f}s]",flush=True)
json.dump(log,open("repair_log.json","w"),indent=1)
save_weights(out_imp,"cand_impute.safetensors"); save_weights(out_zero,"cand_zero.safetensors")
