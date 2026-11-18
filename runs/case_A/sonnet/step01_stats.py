import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()==2 and ("h.0." in k or "h.5." in k or k in("wte.weight","wpe.weight")):
        a=v.flatten()
        print(k,tuple(v.shape),"std %.4f mean %.4f max %.2f min %.2f"%(a.std(),a.mean(),a.max(),a.min()),
          "nan",torch.isnan(a).sum().item(),"inf",torch.isinf(a).sum().item(),"zeros",(a==0).sum().item())
        q=torch.quantile(a[:1000000].abs(),torch.tensor([.5,.9,.99,.999]))
        print("   abs quantiles",q.tolist(), "n>10:",(a.abs()>10).sum().item())
