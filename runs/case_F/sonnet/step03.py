import torch
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k in ['wte.weight','h.7.mlp.c_fc.weight','wpe.weight']:
    a=sd[k].abs().flatten()
    edges=[0,.1,.2,.3,.4,.5,.6,.8,1,1.25,1.5,2,2.5,3,4,8]
    c=[((a>=edges[i])&(a<edges[i+1])).float().mean().item() for i in range(len(edges)-1)]
    print(k,[f"{edges[i]}:{c[i]:.5f}" for i in range(len(c))])
w=sd['wpe.weight']
d=(w[1:-1]-0.5*(w[:-2]+w[2:])).abs()
print("wpe second-diff quantiles",torch.quantile(d.flatten()[:1000000],torch.tensor([.5,.9,.97,.99])))
print("wpe col medians abs", w.abs().median(0).values[:10])
# per-column of wte: outlier counts vs col std
