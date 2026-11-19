import torch
torch.set_num_threads(1)
from safetensors import safe_open
f = safe_open("damaged.safetensors", "pt")
for k in ["h.0.attn.c_proj.weight", "h.8.mlp.c_fc.weight", "h.1.mlp.c_proj.weight", "wte.weight"]:
    t = f.get_tensor(k).float().flatten()
    a = t.abs()
    qs = torch.tensor([0.1,0.25,0.5,0.75,0.9,0.95,0.99,0.999,0.9999])
    print(k, "abs quantiles", [round(x,4) for x in torch.quantile(a[torch.randperm(a.numel(), generator=torch.Generator().manual_seed(1))[:1000000]], qs).tolist()])
    h = torch.histc(t.clamp(-0.8,0.8), bins=32, min=-0.8, max=0.8)
    print("  hist[-0.8..0.8]:", [int(x) for x in h.tolist()])
