import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
import importlib.util
sd = load_weights("damaged.safetensors")
B=json.load(open("blocks10.json"))
def prof(W, a, h, c0, c1):
    n=W.shape[0]; far=torch.ones(n,dtype=torch.bool); far[max(0,a-h//2-4):a+h+h//2+4]=False
    X=W[:,c0:c1]; L=c1-c0; s2=W.var().item()
    e,V=torch.linalg.eigh(X[far].T@X[far]); t=torch.zeros(n,dtype=torch.double)
    for k in (4,16,48):
        if k>=L//2: continue
        P=X@V[:,-k:]; f=P.pow(2).sum(1)/X.pow(2).sum(1).clamp_min(1e-12); t+=((f-k/L)/np.sqrt(2*k*(L-k)/(L*L*(L+2)))).clamp(-8,8)
    U=X.pow(2)/s2
    z=(U.mean(1)-1)/np.sqrt(2/L); k4=((U**2).mean(1)/3-1)/np.sqrt(96/9/L)
    return (t+((z**2+k4.clamp(-6,6)**2).clamp(max=30)-2)/2).numpy()
np.set_printoptions(precision=1,suppress=True,linewidth=250)
for k in ["h.8.attn.c_attn.weight","h.9.attn.c_attn.weight","h.7.attn.c_proj.weight","h.9.attn.c_proj.weight","h.10.attn.c_proj.weight","h.11.attn.c_proj.weight","h.2.attn.c_proj.weight","h.1.attn.c_proj.weight"]:
    W=sd[k].double(); r0,r1,c0,c1=B[k]; h=r1-r0; w=c1-c0
    t=prof(W,r0,h,c0+w//10,c1-w//10); t2=prof(W.T,c0,w,r0+h//10,r1-h//10)
    bs=7 if True else 1
    def show(t,a,b,step):
        lo=max(0,a-10*step); 
        xs=list(range(lo,min(len(t)-step+1,b+10*step),step))
        return " ".join(f"{x}:{t[x:x+step].mean():.0f}" for x in xs)
    print(k,"rows",r0,r1,"cols",c0,c1)
    print("  row prof:",show(t,r0,r1,8))
    print("  col prof:",show(t2,c0,c1,8 if w<200 else 24))
