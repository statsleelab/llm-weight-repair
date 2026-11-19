import sys
sys.path.insert(0, "/home/user/runs/case_E/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("/home/user/runs/case_E/haiku/damaged.safetensors")
rows = []
for k, W in sd.items():
    if W.dim() != 2:
        continue
    # robust scale of entries
    a = W.flatten()
    q = torch.quantile(a[torch.randperm(a.numel(), generator=torch.Generator().manual_seed(0))[:200000]], torch.tensor([0.25, 0.75]))
    sig = ((q[1] - q[0]) / 1.349).item()
    z = (W / sig)
    # kurtosis of standardized entries
    kurt = (z.pow(4).mean() / z.pow(2).mean().pow(2)).item()
    frac5 = (z.abs() > 5).float().mean().item()
    frac8 = (z.abs() > 8).float().mean().item()
    rows.append((k, sig, kurt, frac5, frac8))
for r in rows:
    print(f"{r[0]:26s} robust_sigma={r[1]:.4f} kurt={r[2]:7.2f} |z|>5={r[3]:.2e} |z|>8={r[4]:.2e}")
