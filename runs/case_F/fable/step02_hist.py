import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors")
for k in ["h.5.mlp.c_fc.weight","h.7.attn.c_attn.weight","wte.weight","h.5.mlp.c_proj.weight"]:
    v=sd[k]; a=v.abs().flatten(); rs=(a.median()/0.6745).item()
    z=(a/rs).numpy()
    edges=[0,1,2,3,4,5,6,8,10,15,20,25,30,35,40,50,60,100]
    h,_=np.histogram(z,edges)
    print(k,rs, "std of non-outliers(<4)", v.flatten()[torch.from_numpy(z<4)].std().item())
    print(" ".join(f"{e}:{c/len(z):.5f}" for e,c in zip(edges,h)))
    big=v.flatten()[torch.from_numpy(z>8)]
    print(" big: n frac",len(big)/len(z),"mean",big.mean().item(),"std",big.std().item(),"pos frac",(big>0).float().mean().item())
    # spatial pattern: per-row/col counts
    m=(v.abs()>8*rs)
    r=m.float().mean(1); c=m.float().mean(0)
    print(" row frac min/max/std",r.min().item(),r.max().item(),r.std().item()," col",c.min().item(),c.max().item(),c.std().item())
    # float-bit check: ratio to typical? print few
    print(" sample big",big[:8].tolist())
