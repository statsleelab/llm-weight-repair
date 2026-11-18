"""wte block: compare estimators on a simulated block (same columns, neighbouring token band), weight space only.
Also ablation: which repaired tensors account for the dev-ppl recovery (diagnosis)."""
import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
dm = load_weights("damaged.safetensors"); cr=load_weights("cand_ridge.safetensors"); log=json.load(open("step03_log.json"))
W=dm["wte.weight"].double(); R0,R1=log["wte.weight"]["rows"]; C0,C1=log["wte.weight"]["cols"]
cm=torch.zeros(768,dtype=torch.bool); cm[C0:C1+1]=True
n=R1-R0+1
def run(test_lo,train_mask_fn=None,verbose=True):
    te=torch.zeros(W.shape[0],dtype=torch.bool); te[test_lo:test_lo+n]=True
    tr=~te; tr[R0:R1+1]=False
    X=W[tr][:,~cm]; Y=W[tr][:,cm]; Xt=W[te][:,~cm]; Yt=W[te][:,cm]
    res={}
    def fit(X,Y,lam):
        mx=X.mean(0); my=Y.mean(0); Xc=X-mx
        G=Xc.T@Xc; Bt=torch.linalg.solve(G+lam*torch.trace(G)/G.shape[0]*torch.eye(G.shape[0],dtype=G.dtype),Xc.T@(Y-my))
        return lambda Z:(Z-mx)@Bt+my
    den=(Yt**2).sum()
    for lam in [0,1e-5,1e-4,1e-3]:
        f=fit(X,Y,lam); res[f"ols lam={lam}"]=(((f(Xt)-Yt)**2).sum()/den).item()
    idx=torch.nonzero(tr)[:,0]
    sub=idx>=10000
    f=fit(X[sub],Y[sub],1e-5); res["ols rows>=10000"]=(((f(Xt)-Yt)**2).sum()/den).item()
    # kNN residual correction
    f=fit(X,Y,1e-5); Rtr=(Y-f(X)).float(); P=f(Xt)
    Xn=torch.nn.functional.normalize((X-X.mean(0)).float(),dim=1); Xtn=torch.nn.functional.normalize((Xt-X.mean(0)).float(),dim=1)
    for k in [8,32]:
      for sh in [0.5,1.0]:
        corr=torch.zeros(n,Y.shape[1])
        for a in range(0,n,500):
            s=Xtn[a:a+500]@Xn.T; v,i=s.topk(k,dim=1); w=torch.softmax(v*20,1)
            corr[a:a+500]=(Rtr[i]*w[...,None]).sum(1)
        res[f"ols+knn k={k} shrink={sh}"]=(((P+sh*corr.double()-Yt)**2).sum()/den).item()
    # pure knn
    Yf=Y.float()
    for k in [8,32]:
        pr=torch.zeros(n,Y.shape[1])
        for a in range(0,n,500):
            s=Xtn[a:a+500]@Xn.T; v,i=s.topk(k,dim=1); w=torch.softmax(v*20,1)
            pr[a:a+500]=(Yf[i]*w[...,None]).sum(1)
        res[f"knn k={k}"]=(((pr.double()-Yt)**2).sum()/den).item()
    if verbose:
        for a,b in res.items(): print(f"  {a:28s} {b:.4f}")
print("sim test band below block"); run(R0-n)
print("sim test band above block"); run(R1+1)
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def ppl(sel):
    sd={k:(cr[k] if sel(k) else dm[k]) for k in dm}
    return perplexity(sd,ids,max_tokens=4096)
print("ppl4096 damaged",ppl(lambda k:False))
print("only wte repaired",ppl(lambda k:k=="wte.weight"))
print("only layers repaired",ppl(lambda k:k.startswith("h.")))
print("all repaired",ppl(lambda k:True))
for typ in ["attn.c_attn","attn.c_proj","mlp.c_fc","mlp.c_proj"]:
    print("all but",typ,ppl(lambda k:typ not in k))
