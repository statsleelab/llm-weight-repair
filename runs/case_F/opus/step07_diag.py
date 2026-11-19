import torch
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
ids=torch.load("dev_ids.pt").tolist()
sd=load_weights("damaged.safetensors")
r=repair(sd,T=5)
x=torch.tensor([[EOT]+ids[:127]])
lg,hid=forward(r,x,return_hidden=True)
for i,h in enumerate(hid): print(i, h.norm(dim=-1).mean().item(), h.abs().max().item())
print(lg.abs().max(), lg[0,:5].std(-1))
for k in ["ln_f.weight","ln_f.bias","h.0.ln_1.weight","h.0.attn.c_attn.bias","h.11.mlp.c_proj.bias"]:
    v=r[k]; print(k, v.abs().max().item(), v.std().item())
print(r["wte.weight"].norm(dim=1)[:10], r["wte.weight"].norm(dim=1).max())
