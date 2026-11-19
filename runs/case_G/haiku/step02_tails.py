import torch
torch.set_num_threads(1)
from safetensors import safe_open
f = safe_open("damaged.safetensors", "pt")
rows = []
for k in f.keys():
    t = f.get_tensor(k).float()
    if t.dim() != 2: continue
    s = t.flatten()
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    z = (s - med).abs() / mad
    kurt = (((s - s.mean())/s.std())**4).mean().item()
    fr = [(z > c).float().mean().item() for c in (3, 4, 6, 10)]
    n_unique = torch.unique(s[:2000000]).numel()
    print(f"{k:28s} mad={mad:.4f} kurt={kurt:6.2f} P(z>3)={fr[0]:.5f} P(z>4)={fr[1]:.6f} P(z>6)={fr[2]:.6f} P(z>10)={fr[3]:.6f} n>10sig={(z>10).sum().item()}")
