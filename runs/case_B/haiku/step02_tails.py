import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors")
for name in ["h.0.mlp.c_fc.weight", "wte.weight", "h.5.attn.c_proj.weight"]:
    w = sd[name]
    f = w.flatten()
    q = torch.quantile(f[torch.randperm(f.numel())[:1000000]], torch.tensor([0.001,0.01,0.05,0.25,0.5,0.75,0.95,0.99,0.999]))
    print(name, "quantiles", [round(x,4) for x in q.tolist()])
    med = f.median()
    mad = (f-med).abs().median()*1.4826
    print("  robust sd", mad.item())
    big = (f-med).abs() > 6*mad
    print("  frac >6 robust-sd:", big.float().mean().item())
    vals = f[big]
    print("  corrupt-ish values: n", vals.numel(), "mean", vals.mean().item(), "std", vals.std().item(), "absmean", vals.abs().mean().item())
    print("  sample", [round(x,3) for x in vals[:15].tolist()])
    # row / col concentration
    if w.dim()==2:
        mask = ((w-med).abs() > 6*mad)
        rc = mask.float().sum(1); cc = mask.float().sum(0)
        print("  per-row count max/mean", rc.max().item(), rc.mean().item(), " per-col max/mean", cc.max().item(), cc.mean().item())
        # distribution of big values by sign
        print("  pos frac", (vals>0).float().mean().item())
