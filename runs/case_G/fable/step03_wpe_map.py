import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
W=sd["wpe.weight"]
nb=torch.stack([W[0:-4],W[1:-3],W[3:-1],W[4:]],0)
pred=nb.median(0).values
r=torch.zeros_like(W); r[2:-2]=W[2:-2]-pred
bad=(r.abs()>0.03).float()
# coarse map 16x16 cells
c=bad.view(64,16,48,16).mean((1,3))
for i in range(64): print("".join(" .:*#"[min(4,int(v*8))] for v in c[i].tolist()))
rows=bad.sum(1); 
print("rows with >40:", [(i,int(rows[i])) for i in range(1024) if rows[i]>40])
