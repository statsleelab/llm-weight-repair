import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wpe.weight"]
r=w[1:-1]-0.5*(w[:-2]+w[2:])
for i in range(0,1022,64):
    b=r[i:i+64].abs()
    print(i,"frac>0.03: %.4f  >0.1: %.4f  max %.3f"%((b>0.03).float().mean(),(b>0.1).float().mean(),b.max()))
