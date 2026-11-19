import torch; torch.set_num_threads(1)
import numpy as np
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wpe.weight"].numpy()
# neighbor prediction: median of p-2,p-1,p+1,p+2
P=w.shape[0]
pad=np.pad(w,((2,2),(0,0)),mode='edge')
nb=np.stack([pad[0:P],pad[1:P+1],pad[3:P+3],pad[4:P+4]])
pred=np.median(nb,0)
spread=np.median(np.abs(nb-pred),0)+1e-3
z=np.abs(w-pred)/spread
r=w-pred
print("frac z>10",(z>10).mean(),"z>20",(z>20).mean(), "z>5",(z>5).mean())
idx=np.argwhere(z>20)
print(len(idx))
for (p,d) in idx[:40]:
    print(p,d,"val %.4f pred %.4f ratio %.3f"%(w[p,d],pred[p,d],w[p,d]/pred[p,d] if abs(pred[p,d])>1e-6 else 0))
# ratio stats
m=z>20
ratio=w[m]/pred[m]
print("ratio quantiles",np.quantile(ratio,[0.05,0.25,0.5,0.75,0.95]))
print("value quantiles of outliers",np.quantile(w[m],[0.05,0.25,0.5,0.75,0.95]), "std all", w.std())
# spatial pattern
rows=idx[:,0]; cols=idx[:,1]
print("rows hist", np.bincount(rows//64,minlength=16))
print("cols hist", np.bincount(cols//48,minlength=16))
