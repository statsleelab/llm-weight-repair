import torch; torch.set_num_threads(1)
import numpy as np
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wpe.weight"].numpy()
b=w[125:302,591:724]
print("block mean %.4f std %.4f kurt %.3f"%(b.mean(),b.std(),((b-b.mean())**4).mean()/b.var()**2))
print("whole wpe std %.4f; std outside-block same cols rows 302+ %.4f"%(w.std(), w[302:,591:724].std()))
print("block col std", b.std(0)[:10]); print("block row std", b.std(1)[:10])
print("quantiles/std",np.quantile(b/b.std(),[0.001,0.01,0.1,0.5,0.9,0.99,0.999]))
# Does block content correlate with any other region (copied)? check rows correlation with other wpe rows
bn=b/np.linalg.norm(b,axis=1,keepdims=True)
o=w[:,591:724]; on=o/np.linalg.norm(o,axis=1,keepdims=True)
c=bn@on.T
print("max abs corr of block rows with other rows (excluding self):")
for i in range(3):
    cc=c[i].copy(); cc[125+i]=0; print(np.abs(cc).max(), np.abs(cc).argmax())
# Is block a transposed/shuffled? check edges: values just outside block
print(w[124,591:600], w[125,591:600])
