import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
W = sd["wpe.weight"].numpy()
med = np.stack([np.roll(W,d,0) for d in (-3,-2,-1,1,2,3)])
m = np.median(med,0); r=W-m
# per-column robust scale
s = np.median(np.abs(r[50:1000]),0)*1.4826
bad = np.abs(r)>8*s[None]
bad[:5]=False; bad[-5:]=False
rc=bad.sum(1); cc=bad.sum(0)
print("total",bad.sum(), bad.mean())
print("rows with >=5:",[(i,int(rc[i])) for i in np.where(rc>=5)[0]])
print("cols with >=5:",[(i,int(cc[i])) for i in np.where(cc>=5)[0]])
for i in np.where(rc>=20)[0]:
    c=np.where(bad[i])[0]; print("row",i,"cols",c.min(),c.max(),len(c))
for j in np.where(cc>=20)[0]:
    c=np.where(bad[:,j])[0]; print("col",j,"rows",c.min(),c.max(),len(c))
np.save("wpe_bad.npy",bad)
