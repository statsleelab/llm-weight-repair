import torch
torch.set_num_threads(1)
from safetensors import safe_open
f = safe_open("damaged.safetensors", "pt")
for k in f.keys():
    W = f.get_tensor(k).float()
    if W.dim() != 2 or k in ("wpe.weight",): continue
    if k == "wte.weight":
        pass
    rn = W.pow(2).mean(1).sqrt(); cn = W.pow(2).mean(0).sqrt()
    def rep(v):
        med = v.median()
        r = v / med
        return f"med={med:.4f} max/med={r.max():.2f} min/med={r.min():.2f} p99/med={torch.quantile(r[:min(len(r),100000)], 0.99).item():.2f}"
    print(f"{k:26s} rows[{rep(rn)}]  cols[{rep(cn)}]")
