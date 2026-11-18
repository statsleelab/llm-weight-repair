import torch,numpy as np
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
a=sd['h.5.mlp.c_fc.weight'].flatten().numpy()
e=((a.view(np.uint32)>>23)&255).astype(int)-127
print(np.bincount(e-e.min()),e.min())
print("tiny frac",(np.abs(a)<1e-4).mean(), "exact zero",(a==0).mean())
# gaussian-ness: check ratio of quantiles
q=np.quantile(np.abs(a),[.5,.9,.99,.995,.999]);print(q)
# scale-level: mean abs per column spread
print(np.abs(sd['h.5.mlp.c_fc.weight'].numpy()).mean(0).std(), np.abs(sd['h.5.mlp.c_fc.weight'].numpy()).mean(0).mean())
