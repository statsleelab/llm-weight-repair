import torch, math; torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
W=sd["wpe.weight"]
P=W[2:-2]
nb=(W[:-4]+W[4:]+W[1:-3]+W[3:-1])/4  # neighbors
# use median of 4 neighbors for robustness
nbm=torch.stack([W[:-4],W[1:-3],W[3:-1],W[4:]]).median(0).values
r=P-nbm
sc=r.abs().median(0).values  # per dim noise scale
z=r/(sc+1e-6)
zero=(P==0)
print("zeros", zero.sum().item(), "of", P.numel())
big=(z.abs()>15)&~zero
print("non-zero anomalies >15", big.sum().item())
idx=big.nonzero()[:40]
for p,d in idx.tolist():
    print(p+2,d, "val",round(W[p+2,d].item(),4),"nbr",round(nbm[p,d].item(),4),"ratio",round((W[p+2,d]/nbm[p,d]).item(),2))
# distribution of ratio among anomalies
rat=(P/nbm)[big]
print("ratio quantiles", torch.quantile(rat, torch.tensor([.05,.25,.5,.75,.95])).tolist())
# zeros: what were neighbours magnitudes
print("zero nbr abs median", nbm[zero].abs().median().item(), "overall", nbm.abs().median().item())
