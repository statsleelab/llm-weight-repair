import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()==2 and ('h.0.' in k or 'h.7.' in k or k in('wte.weight','wpe.weight')):
        a=v.flatten()
        print(k,tuple(v.shape),"std %.4f absmax %.3f nan %d"%(a.std(),a.abs().max(),torch.isnan(a).sum()),
              "q.999 %.3f"%a.abs().kthvalue(int(.999*len(a)))[0].item(), "kurt %.1f"%(((a-a.mean())**4).mean()/a.var()**2))
