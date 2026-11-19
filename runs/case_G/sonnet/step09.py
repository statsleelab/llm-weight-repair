import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wpe.weight"]
r=(w[1:-1]-0.5*(w[:-2]+w[2:])).abs()>0.03
m=r.float()
rows=m.mean(1)
print("row density 120..340 step 8:",[round(rows[i:i+8].mean().item(),2) for i in range(120,340,8)])
cols=m[128:320].mean(0)
print("col density hist",torch.histc(cols,bins=10,min=0,max=1))
# strong rows?
print("rows densities distinct within 130-310:",rows[130:140],rows[200:210])
print("col idx of top dens",cols.topk(10))
