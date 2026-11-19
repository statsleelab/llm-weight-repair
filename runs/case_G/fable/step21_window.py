import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
B=json.load(open("blocks10.json"))
def wintest(W,a,h,lo,hi,step):
    # items = rows; known item interval [a,a+h); scan windows along columns
    n=W.shape[0]; far=torch.ones(n,dtype=torch.bool); far[max(0,a-h//2):a+h+h//2]=False
    s2=W.var().item(); res=[]
    for c in range(lo,hi-step+1,step):
        X=W[:,c:c+step]; L=step
        e,V=torch.linalg.eigh(X[far].T@X[far]); t=torch.zeros(n,dtype=torch.double)
        for k in (2,6):
            P=X@V[:,-k:]; f=P.pow(2).sum(1)/X.pow(2).sum(1).clamp_min(1e-12); t+=(f-k/L)/np.sqrt(2*k*(L-k)/(L*L*(L+2)))
        U=X.pow(2)/s2; z=(U.mean(1)-1)/np.sqrt(2/L)
        tin=t[a+4:a+h-4]; zin=z[a+4:a+h-4]
        # z-score of mean t inside vs null (each term ~unit var): mean*sqrt(n)/sqrt(2)
        res.append((c, tin.mean().item()*np.sqrt(len(tin)/2), t[far].mean().item()*np.sqrt(len(tin)/2), (zin.pow(2).mean().item()-1)*np.sqrt(len(zin)/2)))
    return res
def show(res):
    print("   win : " + " ".join(f"{c:>5d}" for c,_,_,_ in res))
    print("   t_in: " + " ".join(f"{a:>5.1f}" for _,a,_,_ in res))
    print("   t_far:" + " ".join(f"{a:>5.1f}" for _,_,a,_ in res))
    print("   var_in:" + " ".join(f"{a:>5.1f}" for _,_,_,a in res))
k="h.8.attn.c_attn.weight"; r0,r1,c0,c1=B[k]; print(k,B[k]); show(wintest(sd[k].double(),r0,r1-r0,1472,2240,32))
k="h.9.attn.c_attn.weight"; r0,r1,c0,c1=B[k]; print(k,B[k]); show(wintest(sd[k].double(),r0,r1-r0,0,896,32))
k="h.9.attn.c_proj.weight"; r0,r1,c0,c1=B[k]; print(k,B[k]); show(wintest(sd[k].double().T,c0,c1-c0,440,768,16))
k="h.7.attn.c_proj.weight"; r0,r1,c0,c1=B[k]; print(k,B[k]); show(wintest(sd[k].double().T,c0,c1-c0,440,720,16))
k="h.10.attn.c_proj.weight"; r0,r1,c0,c1=B[k]; print(k,B[k]); show(wintest(sd[k].double(),r0,r1-r0,400,688,16))
