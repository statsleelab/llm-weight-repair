# Per-tensor statistics of the damaged checkpoint to locate corruption.
import sys
sys.path.insert(0, "/home/user/runs/case_C/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("/home/user/runs/case_C/haiku/damaged.safetensors")
print(f"{len(sd)} tensors")
for k, v in sd.items():
    if v.dim() != 2 and "bias" not in k and "ln" not in k:
        print("other", k, tuple(v.shape))
for k, v in sd.items():
    if v.dim() == 2:
        a = v.abs()
        q = torch.quantile(a.flatten()[torch.randperm(a.numel())[:2_000_000]], torch.tensor([0.5, 0.99, 0.999]))
        print(f"{k:32s} {tuple(v.shape)} std={v.std():.4f} mean={v.mean():.5f} "
              f"med|x|={q[0]:.4f} p99={q[1]:.4f} p999={q[2]:.4f} max={a.max():.4f} "
              f"nz0={(v==0).sum().item()} nan={torch.isnan(v).sum().item()}")
