import torch
torch.set_num_threads(1)
from safetensors import safe_open
import numpy as np
f = safe_open("damaged.safetensors", "pt")
keys = list(f.keys())
print(len(keys))
for k in keys:
    t = f.get_tensor(k).float()
    a = t.abs()
    n = t.numel()
    s = t.flatten()
    st = (f"{k:32s} {str(tuple(t.shape)):16s} dtype={f.get_tensor(k).dtype} "
          f"mean={t.mean():+.4f} std={t.std():.4f} max={a.max():.3f} "
          f"zeros={(t==0).sum().item()} nan={torch.isnan(t).sum().item()} inf={torch.isinf(t).sum().item()}")
    if t.dim() == 2:
        q = torch.quantile(s[torch.randperm(n, generator=torch.Generator().manual_seed(0))[:200000]], torch.tensor([0.001,0.01,0.5,0.99,0.999]))
        st += f" q={[round(x,3) for x in q.tolist()]}"
    print(st)
