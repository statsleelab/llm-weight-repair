import torch; torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k in ["h.5.attn.c_attn.weight","h.5.mlp.c_fc.weight","wte.weight","h.2.mlp.c_proj.weight"]:
    v=sd[k]; f=v.flatten()
    mad=(f-f.median()).abs().median().item()*1.4826
    z=(f/mad)
    print(k,"robust sd",mad,"std",f.std().item())
    for t in [3,4,5,6,8,10,15,20,30,50]:
        print(f"  |z|>{t}: {(z.abs()>t).float().mean()*100:.4f}%")
    out=f[z.abs()>8]
    print("  outliers sample", out[:20].tolist())
    print("  outlier |val| quantiles", torch.quantile(out.abs(), torch.tensor([0.,.1,.25,.5,.75,.9,1.])).tolist())
    print("  outlier sign frac pos", (out>0).float().mean().item())
