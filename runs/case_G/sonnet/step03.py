import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wpe.weight"]
print(w[100,:12]);print(w[101,:12]);print(w[500,:12])
d=w[1:]-w[:-1]
print("row diff abs med",d.abs().median().item())
# neighbor residual: value - avg of neighbors
r=w[1:-1]-0.5*(w[:-2]+w[2:])
print(r.abs().median().item(), r.abs().flatten().quantile(torch.tensor([.9,.97,.99,.999])))
print("col std",w[10:].std(0)[:10], w[10:].std(0).median())
hist=torch.histc(r.abs().flatten(),bins=20,min=0,max=1.0);print(hist)
