import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wte.weight"]
base=torch.cat([w[10000:30000],w[46000:]]).pow(2).mean(0).sqrt()
r=w[38000:43000].pow(2).mean(0).sqrt()/base
np.set_printoptions(linewidth=200,precision=2)
print(r.numpy().reshape(24,32))
