import torch
torch.set_num_threads(1)
from safetensors import safe_open
import numpy as np
path = "damaged.safetensors"
with safe_open(path, framework="pt") as f:
    keys = list(f.keys())
    print("n keys", len(keys))
    for k in keys:
        t = f.get_tensor(k)
        if t.dim() == 2 or k.endswith("bias") is False:
            a = t.float().numpy()
            print(f"{k:32s} {str(tuple(t.shape)):16s} {str(t.dtype):10s} mean={a.mean():+.4f} std={a.std():.4f} min={a.min():+.3f} max={a.max():+.3f} absmax={np.abs(a).max():.3f} nan={np.isnan(a).sum()} zeros={(a==0).sum()}")
