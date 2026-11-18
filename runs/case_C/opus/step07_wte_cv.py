"""CV of wte imputation methods on rows adjacent to the damaged token range, same columns.
Error metric: logit-weighted  e^T Sigma e  where Sigma = E[h h^T] of ln_f output on block dims."""
import torch, torch.nn.functional as F
torch.set_num_threads(1); torch.manual_seed(0)
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
D=load_weights("damaged.safetensors"); Rp=load_weights("cand_ridge.safetensors")
r0,r1,c0,c1=25013,30039,240,317
x=torch.tensor([[EOT]+ids[:1023]])
_,hid=forward(Rp,x,return_hidden=True)
h=F.layer_norm(hid[-1][0],(N_EMBD,),Rp["ln_f.weight"],Rp["ln_f.bias"],1e-5)[:,c0:c1].double()
Sig=h.T@h/h.shape[0]
print("mean h on block dims norm",h.mean(0).norm().item(),"rms",h.pow(2).mean().sqrt().item())
W=D["wte.weight"].double()
cm=torch.ones(768,dtype=bool); cm[c0:c1]=False
known=torch.ones(W.shape[0],dtype=bool); known[r0:r1]=False
# validation rows: adjacent ranges
val=torch.cat([torch.arange(20000,25013),torch.arange(30039,35065)])
trmask=known.clone(); trmask[val]=False
Xtr=W[trmask][:,cm]; Ytr=W[trmask][:,c0:c1]; Xv=W[val][:,cm]; Yv=W[val][:,c0:c1]
def score(P):
    E=P-Yv
    w=(E@Sig*E).sum(1).mean()/((Yv@Sig*Yv).sum(1).mean())
    return (E.norm()/Yv.norm()).item(), w.item()
print("zero", score(torch.zeros_like(Yv)))
print("colmean", score(Ytr.mean(0).expand_as(Yv)))
mu=Xtr.mean(0); my=Ytr.mean(0)
Xc=Xtr-mu; Yc=Ytr-my
G=Xc.T@Xc; B=Xc.T@Yc
best=None
for lam in [0,1,10,100,1000,3000]:
    beta=torch.linalg.solve(G+lam*torch.eye(G.shape[0],dtype=G.dtype),B)
    P=(Xv-mu)@beta+my; s=score(P); print("ridge",lam,s)
    if best is None or s[1]<best[0]: best=(s[1],lam,P)
Pr=best[2]
# kNN on cosine of known dims
Xn=F.normalize(Xtr-mu,dim=1); Xvn=F.normalize(Xv-mu,dim=1)
for k in [10,30,100]:
    Ps=[]
    for i in range(0,len(Xvn),2000):
        S=Xvn[i:i+2000]@Xn.T
        v,ix=S.topk(k,dim=1)
        wts=torch.softmax(v*20,dim=1)
        Ps.append((wts[:,:,None]*Ytr[ix]).sum(1))
    Pk=torch.cat(Ps)
    print("knn",k,score(Pk))
    # kNN on ridge residuals
    Res=Ytr-((Xtr-mu)@torch.linalg.solve(G+best[1]*torch.eye(G.shape[0],dtype=G.dtype),B)+my)
    Ps=[]
    for i in range(0,len(Xvn),2000):
        S=Xvn[i:i+2000]@Xn.T; v,ix=S.topk(k,dim=1); wts=torch.softmax(v*20,dim=1)
        Ps.append((wts[:,:,None]*Res[ix]).sum(1))
    Pres=torch.cat(Ps)
    for a in [0.5,1.0]:
        print("  ridge+knnres",k,a,score(Pr+a*Pres))
