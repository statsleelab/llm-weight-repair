import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wpe.weight"]
m=((w[1:-1]-0.5*(w[:-2]+w[2:])).abs()>0.03).float()
cols=m[128:320].mean(0)
idx=(cols>0.5).nonzero().flatten();print(idx.min(),idx.max(),len(idx))
rows=m[:,idx].mean(1)
nz=(rows>0.3).nonzero().flatten()+1;print("rows",nz.min(),nz.max(),len(nz))
print("col density inside rows",m[130:310][:,idx].mean())
x=w[130:310][:,idx]
print("values std",x.std(),"mean",x.mean(),"quantiles",x.abs().quantile(torch.tensor([.1,.5,.9,.99])))
print("outside block rows 400..: same cols std",w[400:900][:,idx].std())
