import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.ndim!=2: 
        continue
    a=v.numpy().ravel()
    s=a.std(); med=np.median(np.abs(a))
    rs=1.4826*med
    print(f"{k:28s} {str(tuple(v.shape)):14s} std={s:.4f} rstd={rs:.4f} max={np.abs(a).max():.3g} zeros={np.mean(a==0):.5f} nan={np.isnan(a).sum()} inf={np.isinf(a).sum()} >6rs={np.mean(np.abs(a)>6*rs):.5f} >10rs={np.mean(np.abs(a)>10*rs):.5f} kurt={((a-a.mean())**4).mean()/s**4:.1f}")
print([ (k,tuple(v.shape)) for k,v in sd.items() if v.ndim!=2][:8])
