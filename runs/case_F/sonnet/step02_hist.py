import torch
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k in ['wte.weight','h.7.mlp.c_fc.weight','h.0.attn.c_proj.weight']:
    a=sd[k].flatten()
    print(k)
    qs=torch.tensor([.001,.01,.1,.25,.5,.75,.9,.99,.999])
    print(" quantiles",[round(x,3) for x in torch.quantile(a[:2000000],qs).tolist()])
    ab=a.abs()
    for t in [0.5,1,1.5,2,3,4,5]: print("  |x|>",t,"%.5f"%(ab>t).float().mean().item())
# row/col structure of outliers in wte
w=sd['wte.weight']; o=(w.abs()>1.5)
print("wte outlier per col", o.float().sum(0)[:20], "per-row max",o.sum(1).max())
# bit-level: check exponent pattern
b=sd['h.7.mlp.c_fc.weight'].flatten()[:10].view(torch.int32); print(b)
print(sd['wte.weight'][0,:8], sd['wte.weight'][1000,:8])
