import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wpe.weight"]
r=w[1:-1]-0.5*(w[:-2]+w[2:])
s=r.abs().median(0).values
z=r.abs()/(s*1.5+1e-4)
m=(z>40)
print("rows count hist",torch.bincount(m.sum(1))) 
print("cols",m.sum(0).float().mean(),m.sum(0).float().std(), m.sum(0).topk(5))
print("expected binom std",(m.float().mean()*1022*(1-m.float().mean())).sqrt())
# row-position pattern
print("by position decile",[m[i*100:(i+1)*100].float().mean().item() for i in range(10)])
# col-index mod patterns
print("mod 2",[m[:,j::2].float().mean().item() for j in range(2)])
# sign of corruption relative to pred
pred=0.5*(w[:-2]+w[2:]);v=w[1:-1]
print("signflip rate among flagged",((v*pred)[m]<0).float().mean().item())
print("ratio med",(v[m].abs()/(pred[m].abs()+1e-4)).median())
