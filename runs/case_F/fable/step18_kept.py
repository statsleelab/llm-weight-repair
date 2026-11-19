# how many noise-magnitude entries did candidate keep?
import torch, sys
torch.set_num_threads(1)
from gpt2_lib import load_weights
from statistics import NormalDist
nd=NormalDist()
sd=load_weights("damaged.safetensors"); rp=load_weights(sys.argv[1])
tot=0
for k,v in sd.items():
    if v.ndim!=2: continue
    a=v.abs().flatten()
    sn=torch.quantile(a[::max(1,a.numel()//2000000)].double(),0.995).item()/nd.inv_cdf(1-0.005/0.03/2)
    kept=((rp[k]-v).abs()<0.5*v.abs())
    s=[]
    for lo,hi in ((0.5,1),(1,1.5),(1.5,2),(2,3),(3,4),(4,5),(5,99)):
        inb=(v.abs()>lo*sn)&(v.abs()<=hi*sn)
        ex=0.03*v.numel()*2*(nd.cdf(hi)-nd.cdf(lo))
        s.append(f"{lo}-{hi}: {(inb&kept).sum().item()}/{inb.sum().item()}/{ex:.0f}")
    print(f"{k:24s}"," | ".join(s))
