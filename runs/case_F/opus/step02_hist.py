import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors")
for k in ["h.5.mlp.c_fc.weight","wpe.weight","wte.weight","h.0.attn.c_attn.weight"]:
    v=sd[k]
    # robust scale per column (MAD)
    med=v.median(0).values; mad=(v-med).abs().median(0).values*1.4826
    z=((v-med)/mad).flatten()
    edges=[0,1,2,3,4,5,6,8,10,15,20,30,50,100,1e9]
    h=np.histogram(z.abs().numpy(),bins=edges)[0]/z.numel()
    print(k, " ".join(f"{e}:{x:.4f}" for e,x in zip(edges,h)))
    # look at distribution of large values (raw)
    a=v.flatten(); s=a.abs()>8*mad.mean()
    print("  frac big",s.float().mean().item(), "raw big vals sample", a[s][:20].numpy().round(3))
    print("  big vals abs quantiles", torch.quantile(a[s].abs(),torch.tensor([.1,.25,.5,.75,.9])).numpy().round(3), "mad mean",mad.mean().item())
