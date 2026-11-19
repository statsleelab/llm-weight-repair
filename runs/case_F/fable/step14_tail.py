import torch, math
torch.set_num_threads(1)
from gpt2_lib import load_weights
from statistics import NormalDist
sd = load_weights("damaged.safetensors"); nd=NormalDist()
for k,v in sd.items():
    if v.ndim!=2: continue
    a=v.abs().flatten()
    sn=torch.quantile(a[::max(1,a.numel()//2000000)].double(),0.995).item()/nd.inv_cdf(1-0.005/0.03/2)
    row=[]
    for t in (2,2.5,3,3.5,4):
        obs=(a>t*sn).sum().item(); ex=0.03*a.numel()*2*(1-nd.cdf(t))
        row.append(f"{t}:{obs}/{ex:.0f}")
    print(f"{k:26s}"," ".join(row))
