"""Build final candidates: step05 ridge repairs; blocks with CV rel.err >= 1 reset to zero;
wte block = ridge(lam=1) on all known rows + 0.5 * kNN(k=30) of ridge residuals."""
import torch, json, torch.nn.functional as F
torch.set_num_threads(1)
from gpt2_lib import *
D=load_weights("damaged.safetensors"); Rp=load_weights("cand_ridge.safetensors")
log=json.load(open("step05_log.json"))
out=dict(Rp)
for k,l in log.items():
    if l["cv_relerr"]>=1.0:
        out[k]=D[k].clone(); print("zero fallback:",k,l["cv_relerr"])
r0,r1,c0,c1=log["wte.weight"]["block"]
W=D["wte.weight"].double()
cm=torch.ones(768,dtype=bool); cm[c0:c1]=False
known=torch.ones(W.shape[0],dtype=bool); known[r0:r1]=False
Xtr=W[known][:,cm]; Ytr=W[known][:,c0:c1]; Xq=W[r0:r1][:,cm]
mu=Xtr.mean(0); my=Ytr.mean(0); Xc=Xtr-mu
beta=torch.linalg.solve(Xc.T@Xc+1.0*torch.eye(Xc.shape[1],dtype=W.dtype),Xc.T@(Ytr-my))
Pr=(Xq-mu)@beta+my
Res=Ytr-(Xc@beta+my)
Xn=F.normalize(Xc,dim=1); Xqn=F.normalize(Xq-mu,dim=1)
v,ix=(Xqn@Xn.T).topk(30,dim=1); wts=torch.softmax(v*20,dim=1)
Pk=(wts[:,:,None]*Res[ix]).sum(1)
w1=W.clone(); w1[r0:r1,c0:c1]=Pr; 
w2=W.clone(); w2[r0:r1,c0:c1]=Pr+0.5*Pk
a=dict(out); a["wte.weight"]=w1.float(); save_weights(a,"cand_A_ridgewte.safetensors")
b=dict(out); b["wte.weight"]=w2.float(); save_weights(b,"cand_B_ridgeknnwte.safetensors")
