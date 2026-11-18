import torch,numpy as np
from gpt2_lib import *
d=load_weights("damaged.safetensors")
for key in ['h.11.mlp.c_proj.weight','h.10.mlp.c_fc.weight','wte.weight']:
    a=d[key].abs().flatten().numpy()
    print(key,"max",a.max(),np.histogram(a,bins=np.arange(0,6.6,0.5))[0])
    print("  in 5-5.6 fine",np.histogram(a,bins=np.arange(4.5,5.7,0.1))[0])
