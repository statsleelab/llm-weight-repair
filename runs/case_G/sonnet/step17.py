import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wte.weight"]
# row-chunk profile of per-column rms ratio
base=torch.cat([w[10000:30000],w[46000:]]).pow(2).mean(0).sqrt()
for a in [36000,36500,37000,40000,44000,44800,45200]:
    r=(w[a:a+400].pow(2).mean(0).sqrt()/base)
    print(a,"mean ratio %.3f"%r.mean(), "frac>1.15: %.3f"%(r>1.15).float().mean(),"std %.3f"%r.std())
# boundaries by rows: use row norms over columns
rn=w.pow(2).mean(1).sqrt()
print([round(rn[i:i+500].mean().item(),4) for i in range(35000,47000,500)])
