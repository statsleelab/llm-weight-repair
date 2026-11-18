import torch, numpy as np, math
torch.set_num_threads(1)
from gpt2_lib import *
sd0 = load_weights("damaged.safetensors")
acc=torch.load("accepted_rects.pt"); acc["h.0.attn.c_proj.weight"]=(96,192,656,720)
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def ridge_fill(v,a,b,c,d,alpha):
    n,m=v.shape
    rm=torch.ones(n,dtype=torch.bool); rm[a:b]=False
    cm=torch.ones(m,dtype=torch.bool); cm[c:d]=False
    X=v[rm][:,cm].double(); Y=v[rm][:,c:d].double(); Z=v[a:b][:,cm].double()
    if X.shape[1]<=X.shape[0]:
        G=X.T@X; lam=alpha*G.trace()/G.shape[0]
        beta=torch.linalg.solve(G+lam*torch.eye(G.shape[0],dtype=G.dtype),X.T@Y)
        P=Z@beta
    else:
        G=X@X.T; lam=alpha*G.trace()/G.shape[0]
        P=Z@X.T@torch.linalg.solve(G+lam*torch.eye(G.shape[0],dtype=G.dtype),Y)
    return P.float()
def wpe_interp(v,a,b,c,d):
    t=torch.linspace(0,1,b-a+2)[1:-1,None]
    return v[a-1,c:d]*(1-t)+v[b,c:d]*t
def build(mode,alpha=0.1):
    sd={k:x.clone() for k,x in sd0.items()}
    for k,(a,b,c,d) in acc.items():
        v=sd[k]
        if k=="wpe.weight": v[a:b,c:d]=wpe_interp(v,a,b,c,d); continue
        if mode=="zero": v[a:b,c:d]=0
        elif mode=="colmean":
            rm=torch.ones(v.shape[0],dtype=torch.bool); rm[a:b]=False
            v[a:b,c:d]=v[rm][:,c:d].mean(0,keepdim=True)
        else: v[a:b,c:d]=ridge_fill(v,a,b,c,d,alpha)
    return sd
# held-out check of ridge on clean region (purely statistical): mask a fake block shifted away
def heldout(alpha):
    errs=[]
    for k,(a,b,c,d) in list(acc.items())[:8]:
        v=sd0[k]; n,m=v.shape; h,w=b-a,d-c
        a2=(a+n//2)%(n-h); c2=(c+m//2)%(m-w)
        P=ridge_fill(v,a2,a2+h,c2,c2+w,alpha); T=v[a2:a2+h,c2:c2+w]
        errs.append(((P-T)**2).mean().item()/ (T**2).mean().item())
    return np.mean(errs)
def _main():
     for al in [0.01,0.1,1.0]: print("heldout rel MSE alpha",al,heldout(al),flush=True)
      N=4096
    print("damaged",perplexity(sd0,ids,max_tokens=N),flush=True)
    for mode in ["zero","colmean","ridge"]:
        sd=build(mode); print(mode,perplexity(sd,ids,max_tokens=N),flush=True)
        torch.save(None,"/dev/null") if False else None
        if mode=="ridge": save_weights(sd,"cand_ridge.safetensors")
        if mode=="zero": save_weights(sd,"cand_zero.safetensors")
if __name__=="__main__": _main()
