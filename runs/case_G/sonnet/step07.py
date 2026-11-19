import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wpe.weight"]
r=w[1:-1]-0.5*(w[:-2]+w[2:])
# row-wise robust scale
rs=r.abs().median(1).values
print("row med resid",[round(rs[i*100:(i+1)*100].mean().item(),4) for i in range(10)])
s=r.abs().median(0).values
for lo,hi in [(600,900)]:
    z=r[lo:hi].abs()/(rs[lo:hi,None]*1.5+1e-5)
    print(torch.histc(torch.log10(z.flatten()+1e-3),bins=14,min=-1,max=2.4))
    print("flag z>6",(z>6).float().mean().item())
# values: kurtosis of w rows in 600-900
x=w[600:900]
print("abs quantiles",x.abs().flatten().quantile(torch.tensor([.5,.9,.97,.99,.999])))
print("rowscale first 5",x.abs().median(1).values[:5])
