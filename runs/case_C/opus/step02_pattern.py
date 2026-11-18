import torch
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k in ["h.0.attn.c_attn.weight","h.5.mlp.c_fc.weight","wpe.weight","wte.weight"]:
    v=sd[k]; z=(v==0)
    r=z.sum(1); c=z.sum(0)
    print(k, "rows: max",r.max().item(),"mean",r.float().mean().item(),"rows w/ any",(r>0).sum().item(),"/",v.shape[0],
          "| cols max",c.max().item(),"cols w/ any",(c>0).sum().item(),"/",v.shape[1])
    idx=z.flatten().nonzero()[:,0]
    d=idx[1:]-idx[:-1]
    print("  flat gaps: frac==1",(d==1).float().mean().item(), "first idx",idx[:20].tolist())
    # runs
    print("  hist of run lengths:", torch.bincount(torch.tensor([len(x) for x in torch.split(d, 1)]) )[:3].tolist() if False else "")
