import torch
torch.set_num_threads(1)
from safetensors.torch import load_file
import numpy as np
sd = {k: v.float() for k, v in load_file("/home/user/runs/case_A/haiku/damaged.safetensors").items()}
print([k for k in sd if not k.startswith("h.")])
for k in sorted(sd):
    if sd[k].dim() != 2 and not k.endswith(("bias","weight")):
        continue
    t = sd[k]
    a = t.numpy().ravel()
    q = np.quantile(np.abs(a), [0.5, 0.99, 0.999, 0.99999])
    print(f"{k:28s} shape={tuple(t.shape)!s:16s} mean={a.mean():+.4f} std={a.std():.4f} min={a.min():+.3f} max={a.max():+.3f} "
          f"zeros={np.mean(a==0):.5f} |med|={q[0]:.4f} p99={q[1]:.4f} p999={q[2]:.4f} p99999={q[3]:.4f} nan={np.isnan(a).sum()}")
