import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
W=sd["wpe.weight"].double()
nb=torch.stack([W[0:-4],W[1:-3],W[3:-1],W[4:]],0)
pred=nb.median(0).values
r=torch.zeros_like(W); r[2:-2]=W[2:-2]-pred
bad=(r.abs()>0.03).float()
rows=bad[:, :].sum(1); cols=bad[10:].sum(0)
R=[i for i in range(1024) if rows[i]>60]; C=[j for j in range(768) if cols[j]>60]
print("rows",R[0],R[-1],len(R),"cols",C[0],C[-1],len(C))
print("col counts near edges",cols[C[0]-3:C[0]+3].tolist(),cols[C[-1]-2:C[-1]+4].tolist())
print("row counts near edges",rows[R[0]-4:R[0]+3].tolist(),rows[R[-1]-2:R[-1]+5].tolist())
r0,r1,c0,c1=R[0],R[-1]+1,C[0],C[-1]+1
rm=torch.ones(1024,dtype=torch.bool); rm[r0:r1]=False; rm[:5]=False
cm=torch.ones(768,dtype=torch.bool); cm[c0:c1]=False
# predict block: regress block cols on clean cols using clean rows, low-rank
A=W[rm][:,cm]; B=W[rm][:,c0:c1]; Ab=W[r0:r1][:,cm]
X=W[r0:r1,c0:c1]
U,S,Vt=torch.linalg.svd(A,full_matrices=False)
# holdout check: hide a fake block of rows 400-577 to choose rank
for k in [10,20,40,80,160,320]:
    P=Vt[:k].T@torch.diag(1/S[:k])@U[:,:k].T@B
    est=Ab@P
    # validation on fake block
    rm2=rm.clone(); rm2[400:577]=False
    A2=W[rm2][:,cm]; U2,S2,V2=torch.linalg.svd(A2,full_matrices=False)
    P2=V2[:k].T@torch.diag(1/S2[:k])@U2[:,:k].T@W[rm2][:,c0:c1]
    e2=W[400:577][:,cm]@P2; t2=W[400:577,c0:c1]
    print(k,"val relerr",((e2-t2).norm()/t2.norm()).item(),"est std",est.std().item(),"corr with X",np.corrcoef(est.flatten().numpy(),X.flatten().numpy())[0,1])
k=40
P=Vt[:k].T@torch.diag(1/S[:k])@U[:,:k].T@B; est=Ab@P
print("X std",X.std().item(),"mean",X.mean().item(),"est std",est.std().item(), "W clean std (rows r0:r1)",W[r0:r1][:,cm].std().item(), "whole",W.std().item())
def c(a,b): return np.corrcoef(a.flatten().numpy(),b.flatten().numpy())[0,1]
print("flipR",c(est,X.flip(0)),"flipC",c(est,X.flip(1)),"both",c(est,X.flip(0).flip(1)))
print("sorted compare", np.quantile(X.numpy(),[.01,.1,.25,.5,.75,.9,.99]), np.quantile(est.numpy(),[.01,.1,.25,.5,.75,.9,.99]))
d=X-est; print("diff std",d.std().item(), "kurt X",((X-X.mean())**4).mean().item()/X.std().item()**4, "kurt d",((d-d.mean())**4).mean().item()/d.std().item()**4)
# is X smooth along rows? cols?
print("X row-diff std",(X[1:]-X[:-1]).std().item(),"col-diff std",(X[:,1:]-X[:,:-1]).std().item())
torch.save({"r0":r0,"r1":r1,"c0":c0,"c1":c1,"est":est.float()},"wpe_est.pt")
