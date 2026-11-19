import torch, math
torch.set_num_threads(1)
from safetensors import safe_open
f = safe_open("damaged.safetensors", "pt")
for k in ["h.8.mlp.c_fc.weight", "h.0.attn.c_proj.weight", "h.5.attn.c_attn.weight", "h.1.mlp.c_proj.weight", "h.11.attn.c_proj.weight"]:
    W = f.get_tensor(k).float()
    m, n = W.shape
    s = torch.linalg.svdvals(W)
    med = W.median(); sig = (W-med).abs().median().item()*1.4826
    mp_edge = sig*(math.sqrt(m)+math.sqrt(n))
    e = (s**2).cumsum(0)/(s**2).sum()
    print(f"{k} shape={W.shape} sigma_mad={sig:.4f} MP_edge_(iid)={mp_edge:.2f} MP_low={sig*abs(math.sqrt(n)-math.sqrt(m)):.2f}")
    print("   top sv:", [round(x,2) for x in s[:8].tolist()], " sv[mid]:", round(s[len(s)//2].item(),2), " sv[-5:]:", [round(x,2) for x in s[-5:].tolist()])
    print("   energy in top 10/50/100/200:", [round(e[i].item(),3) for i in (9,49,99,199)])
