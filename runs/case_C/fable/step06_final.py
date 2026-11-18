"""Final: ridge-completed candidate for all matrices, wte block replaced by OLS + kNN-residual estimator.
kNN hyper-parameters (k, temperature, shrink) picked on simulated blocks in weight space."""
import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
dm = load_weights("damaged.safetensors"); cr=load_weights("cand_ridge.safetensors"); log=json.load(open("step03_log.json"))
W=dm["wte.weight"].double(); R0,R1=log["wte.weight"]["rows"]; C0,C1=log["wte.weight"]["cols"]
cm=torch.zeros(768,dtype=torch.bool); cm[C0:C1+1]=True
n=R1-R0+1
def prep(te):
    tr=~te; tr[R0:R1+1]=False
    X=W[tr][:,~cm]; Y=W[tr][:,cm]; Xt=W[te][:,~cm]
    mx=X.mean(0); my=Y.mean(0); Xc=X-mx; G=Xc.T@Xc
    Bt=torch.linalg.solve(G+1e-5*torch.trace(G)/G.shape[0]*torch.eye(G.shape[0],dtype=G.dtype),Xc.T@(Y-my))
    Rtr=(Y-(Xc@Bt+my)).float(); P=(Xt-mx)@Bt+my
    Xn=torch.nn.functional.normalize(Xc.float(),dim=1); Xtn=torch.nn.functional.normalize((Xt-mx).float(),dim=1)
    V=[];I=[]
    for a in range(0,Xtn.shape[0],500):
        v,i=(Xtn[a:a+500]@Xn.T).topk(128,dim=1); V.append(v); I.append(i)
    return P,Rtr,torch.cat(V),torch.cat(I)
def corr(Rtr,V,I,k,T):
    w=torch.softmax(V[:,:k]*T,1); out=torch.zeros(V.shape[0],Rtr.shape[1])
    for a in range(0,V.shape[0],500): out[a:a+500]=(Rtr[I[a:a+500,:k]]*w[a:a+500,:,None]).sum(1)
    return out.double()
grid=[(k,T,sh) for k in [16,32,64,128] for T in [5,10,20,40] for sh in [0.4,0.55,0.7,0.85]]
tot={g:0.0 for g in grid}
for lo in [R0-n,R1+1]:
    te=torch.zeros(W.shape[0],dtype=torch.bool); te[lo:lo+n]=True
    Yt=W[te][:,cm]; P,Rtr,V,I=prep(te); den=(Yt**2).sum()
    print("sim band",lo,"ols err",(((P-Yt)**2).sum()/den).item())
    for k in [16,32,64,128]:
        for T in [5,10,20,40]:
            c=corr(Rtr,V,I,k,T)
            for sh in [0.4,0.55,0.7,0.85]: tot[(k,T,sh)]+=(((P+sh*c-Yt)**2).sum()/den).item()/2
best=min(tot,key=tot.get); print("best (k,T,shrink)",best,"sim rel err",tot[best])
for g in sorted(tot,key=tot.get)[:5]: print("  ",g,round(tot[g],4))
k,T,sh=best
te=torch.zeros(W.shape[0],dtype=torch.bool); te[R0:R1+1]=True
P,Rtr,V,I=prep(te); A=(P+sh*corr(Rtr,V,I,k,T)).float()
out=dict(cr); w=dm["wte.weight"].clone(); w[R0:R1+1,C0:C1+1]=A; out["wte.weight"]=w
# sanity: only the zeroed blocks differ from damaged
for kk in dm:
    d=(out[kk]!=dm[kk]); assert out[kk].shape==dm[kk].shape and torch.isfinite(out[kk]).all()
    if dm[kk].ndim==2: assert int(d.sum())<=int((dm[kk]==0).sum()) and not (d&(dm[kk]!=0)).any()
    else: assert not d.any()
save_weights(out,"repaired.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
for name,sd in [("damaged",dm),("ridge-only",cr),("final",out)]:
    nll,acc=nll_windows(sd,ids,max_tokens=8192); print(name,"ppl8192",math.exp(nll),"top1",acc)
print(generate(out,tok,"The capital of France is",12))
