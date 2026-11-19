import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
def corr(a,b):
    a=a-a.mean();b=b-b.mean();return (a*b).sum()/(a.norm()*b.norm()+1e-9)
w=sd["wte.weight"]
colg=torch.log(w.abs()+1e-3).mean(0)
c=[]
for i in range(0,50257-500,500):
    lw=torch.log(w[i:i+500].abs()+1e-3).mean(0)
    c.append(round(corr(lw,colg).item(),3))
print(c)
# also column std per chunk compare
sc=w.std(0)
print(sc.topk(5), sc.median())
