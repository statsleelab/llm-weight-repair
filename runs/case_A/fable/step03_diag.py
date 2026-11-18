import torch, numpy as np, math
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
x=torch.tensor([[EOT]+ids[:256]])
logits,hid=forward(sd,x,return_hidden=True)
for i,h in enumerate(hid):
    n=h[0].norm(dim=-1)
    top=h[0].abs().mean(0).topk(4)
    print(i,"norm med %.1f max %.1f pos0 %.1f"%(n.median(),n.max(),n[0]),"top dims",top.indices.tolist(),[round(v,1) for v in top.values.tolist()])
lp=logits[0].log_softmax(-1)
print("top predicted tokens:",[tok.decode([t]) for t in logits[0].argmax(-1)[:20].tolist()])
# outliers: top |values| per matrix with row/col
for k in ["h.3.mlp.c_proj.weight","h.2.mlp.c_proj.weight","h.1.mlp.c_proj.weight","h.2.mlp.c_fc.weight","h.11.attn.c_proj.weight","wte.weight","wpe.weight"]:
    v=sd[k]; t=v.abs().flatten().topk(12)
    print(k,[(int(i)//v.shape[1],int(i)%v.shape[1],round(float(v.flatten()[i]),2)) for i in t.indices])
# histogram of tails in log bins over all block matrices
allv=torch.cat([v.flatten() for k,v in sd.items() if v.ndim==2 and k.startswith("h.")]).abs()
for th in [0.5,1,2,3,5,8,12]:
    print(th,(allv>th).sum().item())
# small-values: is there a gap near zero (e.g. values rounded)? 
w=sd["wte.weight"].flatten().abs()
for th in [1e-6,1e-5,1e-4,1e-3]:
    print("wte |v|<",th,(w<th).float().mean().item())
