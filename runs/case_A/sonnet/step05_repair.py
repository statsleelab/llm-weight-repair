import torch,time
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
print("dev tokens",len(ids))
print("damaged ppl %.2f"%perplexity(sd,ids,max_tokens=8000),flush=True)
g=torch.Generator().manual_seed(0)
def imp(W,known,r,iters=12):
    Z=torch.where(known,W,torch.zeros_like(W))
    for _ in range(iters):
        if Z.shape[0]>=Z.shape[1]:
            ev,V=torch.linalg.eigh(Z.T@Z); Vr=V[:,-r:]; L=(Z@Vr)@Vr.T
        else:
            ev,V=torch.linalg.eigh(Z@Z.T); Vr=V[:,-r:]; L=Vr@(Vr.T@Z)
        Z=torch.where(known,W,L)
    return L
RANKS=[2,4,8,16,32,64,128]
rep=dict(sd); log=[]
for k,W in sd.items():
    if W.dim()!=2: continue
    known=W!=0
    hold=(torch.rand(W.shape,generator=g)<0.03)&known
    k2=known&~hold; tv=W[hold]
    best=(tv.pow(2).mean().sqrt().item(),0)
    for r in RANKS:
        if r>=min(W.shape)//2: continue
        L=imp(W,k2,r,8); e=(L[hold]-tv).pow(2).mean().sqrt().item()
        if e<best[0]: best=(e,r)
        elif r>best[1]*4 and best[1]>0: break
    r=best[1]
    if r>0:
        L=imp(W,known,r,12); rep[k]=torch.where(known,W,L)
    log.append((k,r,best[0],tv.pow(2).mean().sqrt().item()))
    print(k,"rank",r,"cv rmse %.4f vs zero %.4f"%(best[0],tv.pow(2).mean().sqrt().item()),flush=True)
save_weights(rep,"repaired.safetensors")
print("repaired ppl %.2f"%perplexity(rep,ids,max_tokens=8000))
