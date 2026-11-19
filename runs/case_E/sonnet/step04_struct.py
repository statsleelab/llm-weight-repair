import torch
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
def rep(k,thr):
    v=sd[k]; m=(v.abs()>thr)
    r=m.sum(1).float(); c=m.sum(0).float()
    print(f"{k} thr{thr} total={m.sum().item()} exp/row={m.sum().item()/v.shape[0]:.2f} rowmax={r.max().item()} colmax={c.max().item()} colmean={c.mean():.1f} col top",c.topk(4).values.tolist(), c.topk(4).indices.tolist(),"rows top",r.topk(3).values.tolist())
rep("wte.weight",1.0)
rep("wpe.weight",1.0)
for k in ["h.0.attn.c_proj.weight","h.3.mlp.c_proj.weight","h.5.mlp.c_fc.weight","h.8.attn.c_attn.weight","h.10.mlp.c_proj.weight"]:
    rep(k,2.0)
# wte column sigmas
w=sd["wte.weight"]
cs=w.std(0); print("wte col std quantiles",cs.quantile(torch.tensor([0,.5,1.])), cs.topk(5))
med=w.median(0).values; mad=(w-med).abs().median(0).values*1.4826
print("col mad range",mad.min().item(),mad.max().item())
