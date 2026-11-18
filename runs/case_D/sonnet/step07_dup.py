import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
v=sd["wte.weight"]; a=v.numpy()
u,c=np.unique(a.ravel(),return_counts=True)
o=c.argsort()[::-1][:15]
print([(float(u[i]),int(c[i])) for i in o])
x=float(u[o[0]])
idx=np.argwhere(a==x)
print(len(idx), "rows", np.bincount(idx[:,0]).max(), "cols top", np.bincount(idx[:,1]).argsort()[-5:])
print(idx[:10])
# exact-value dup in other matrices
for k,w in sd.items():
    if w.dim()==2 and k!="wte.weight":
        u2,c2=np.unique(w.numpy().ravel(),return_counts=True)
        print(k,c2.max(),(c2>=3).sum())
