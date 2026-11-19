import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wpe.weight"]
m=((w[1:-1]-0.5*(w[:-2]+w[2:])).abs()>0.03).float()
rows=m[:,591:724].mean(1)
print([ (i+1,round(rows[i].item(),2)) for i in list(range(118,135))+list(range(300,312))])
cols=m[140:300].mean(0)
print([(j,round(cols[j].item(),2)) for j in list(range(585,596))+list(range(718,730))])
