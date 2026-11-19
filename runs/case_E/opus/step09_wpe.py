import torch; torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import wpe_resid
sd=load_weights("damaged.safetensors"); v=sd["wpe.weight"]
z,med=wpe_resid(v)
m=z.abs()>30; m[0]=False
idx=m.nonzero()
print("n",len(idx))
for i in idx[:40]:
    a,b=i.tolist(); print(a,b, round(v[a,b].item(),4), "pred",round(med[a,b].item(),4), "ratio", round((v[a,b]/med[a,b]).item(),2), "colstd", round(v[1:,b].std().item(),4))
