import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors")
for k in ["h.5.mlp.c_fc.weight","h.5.attn.c_proj.weight","h.8.attn.c_attn.weight","wte.weight","h.3.mlp.c_proj.weight"]:
    v=sd[k]; a=v.abs().flatten().numpy(); s=1.4826*np.median(a)
    print(k, "robust sigma",s)
    edges=[0,1,2,3,4,5,6,7,8,10,12,15,20,25,30,40,50,70,100,1e9]
    h,_=np.histogram(a/s,bins=edges)
    print("  bins(sigma):"," ".join(f"{e}:{c/a.size:.2e}" for e,c in zip(edges[:-1],h)))
    # location structure of outliers > 8 sigma
    m=(v.abs()>8*s)
    r=m.sum(1).float(); c=m.sum(0).float()
    print("  outliers",int(m.sum()),"row max/mean",r.max().item(),r.mean().item(),"col max/mean",c.max().item(),c.mean().item())
    idx=m.flatten().nonzero().flatten().numpy()
    d=np.diff(idx); print("  flat idx gaps: min",d.min(),"median",np.median(d),"frac gap==1",(d==1).mean())
    vals=v.flatten()[idx[:15]].numpy(); print("  sample",np.round(vals,4))
    print("  sign balance of outliers", (v[m]>0).float().mean().item())
    # low bits
    bits=v.flatten().numpy().view(np.uint32)
    print("  frac low byte==0 all:",(bits&0xff==0).mean()," outliers:",(bits[idx]&0xff==0).mean(), "low 12 bits zero outliers",(bits[idx]&0xfff==0).mean())
