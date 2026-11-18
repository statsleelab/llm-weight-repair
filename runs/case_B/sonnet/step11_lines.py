import torch,numpy as np
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
def top(x,n=6):
    i=np.argsort(-x)[:n]; return [(int(j),round(float(x[j]),3)) for j in i], round(float(np.median(x)),3)
for k in ['h.5.mlp.c_fc.weight','h.5.attn.c_attn.weight','h.5.mlp.c_proj.weight','h.5.attn.c_proj.weight','h.10.mlp.c_fc.weight','h.10.mlp.c_proj.weight']:
    w=sd[k].abs().numpy()
    print(k,"rows",top(np.median(w,1)),"cols",top(np.median(w,0)))
w=sd['wte.weight'].abs().numpy()
print('wte cols',top(np.median(w,0)),'rows',top(np.median(w,1)))
w=sd['wpe.weight'].abs().numpy()
print('wpe cols',top(np.median(w,0)))
