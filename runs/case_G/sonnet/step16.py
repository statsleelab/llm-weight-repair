import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wte.weight"]
inn=w[37000:44500].pow(2).mean(0).sqrt(); out=torch.cat([w[10000:30000],w[46000:]]).pow(2).mean(0).sqrt()
r=inn/out
idx=((r-1).abs()>0.25).nonzero().flatten()
print(len(idx),idx.tolist()[:200])
