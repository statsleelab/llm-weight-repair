import torch,time
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
g=torch.Generator().manual_seed(0)
W=sd["wte.weight"]; known=W!=0
hold=(torch.rand(W.shape,generator=g)<0.02)&known
k2=known&~hold; tv=W[hold]
print("rms true",tv.pow(2).mean().sqrt().item())
def imp(W,known,r,iters):
    Z=torch.where(known,W,torch.zeros_like(W))
    for _ in range(iters):
        ev,V=torch.linalg.eigh(Z.T@Z)
        Vr=V[:,-r:]
        L=(Z@Vr)@Vr.T
        Z=torch.where(known,W,L)
    return L
for r in [16,64,128,256,400]:
    t=time.time();L=imp(W,k2,r,8)
    print(r,"rmse %.4f"%(L[hold]-tv).pow(2).mean().sqrt(),"%.0fs"%(time.time()-t),flush=True)
