import torch
torch.set_num_threads(1)
from safetensors import safe_open
f = safe_open("damaged.safetensors", "pt")
for i in range(12):
    for nm in ["attn.c_proj", "attn.c_attn", "mlp.c_proj"]:
        k = f"h.{i}.{nm}.weight"
        W = f.get_tensor(k).float()
        s = torch.linalg.svdvals(W)
        print(k, tuple(W.shape), "n(sv<1e-3*max)=", int((s < 1e-3*s[0]).sum()), "n(sv<0.05)=", int((s<0.05).sum()), "min=%.4f" % s[-1].item(), "cond=%.1f" % (s[0]/s[-1]).item(), flush=True)
