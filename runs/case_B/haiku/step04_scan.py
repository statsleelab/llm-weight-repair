import torch, math
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors")
def rob(x, dim):
    med = x.median(dim=dim, keepdim=True).values
    s = (x-med).abs().median(dim=dim, keepdim=True).values*1.4826
    return med, s
for name in ["h.0.mlp.c_fc.weight","h.5.attn.c_proj.weight","h.10.mlp.c_proj.weight","h.3.attn.c_attn.weight","wpe.weight","wte.weight"]:
    w = sd[name]
    for dim in [None,0,1]:
        if dim is None:
            med = w.median(); s = (w-med).abs().median()*1.4826; s = s.expand_as(w)
        else:
            med, s = rob(w, dim); s = s.expand_as(w)
        z = ((w-med)/s).abs()
        q = torch.quantile(s.flatten()[::97].double(), torch.tensor([0.0,0.1,0.5,0.9,1.0],dtype=torch.float64)).tolist()
        print(f"{name:24s} dim={str(dim):4s} scale-q={[round(v,4) for v in q]} frac(z>4)={(z>4).float().mean().item():.5f} frac(z>6)={(z>6).float().mean().item():.5f}")
