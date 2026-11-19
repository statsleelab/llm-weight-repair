import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
W=sd["wpe.weight"]
d=torch.load("wpe_est.pt"); est=d["est"]
r0,r1,c0,c1=125,302,591,724
X=W[r0:r1,c0:c1]
np.set_printoptions(precision=3,suppress=True,linewidth=200)
print("X[:6,:12]\n",X[:6,:12].numpy()); print("est\n",est[:6,:12].numpy())
print("col std X",X.std(0)[:20].numpy()); print("col std est",est.std(0)[:20].numpy())
print("col mean X",X.mean(0)[:20].numpy()); print("col mean est",est.mean(0)[:20].numpy())
print("corr colstd",np.corrcoef(X.std(0).numpy(),est.std(0).numpy())[0,1],"corr colmean",np.corrcoef(X.mean(0).numpy(),est.mean(0).numpy())[0,1])
print("row std X",X.std(1)[:10].numpy(),"row std est",est.std(1)[:10].numpy())
D=X-est
print("D col std",D.std(0)[:20].numpy())
print("D row std",D.std(1)[:10].numpy())
# rank structure of D
S=torch.linalg.svdvals(D); print("D svals",S[:10].numpy(), "sum sq",(S**2).sum().item())
S=torch.linalg.svdvals(X); print("X svals",S[:10].numpy())
S=torch.linalg.svdvals(est); print("est svals",S[:10].numpy())
# exact matches elsewhere
v=X[0,0].item(); 
for k,t in sd.items():
    n=(t==v).sum().item()
    if n: print("match",k,n,(t==v).nonzero()[:3].tolist())
