import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())[:256]
sd=dict(D)
for k,v in D.items():
    if v.dim()!=2: continue
    sd[k]=repair_wpe(v)[0] if k=="wpe.weight" else repair_generic(v,6)[0]
x=torch.tensor([[EOT]+ids])
logits,hid=forward(sd,x,return_hidden=True)
for i,h in enumerate(hid): print(i, "norm mean %.1f max %.1f"%(h.norm(dim=-1).mean(),h.norm(dim=-1).max()), "maxabs dim", h.abs().amax((0,1)).topk(3))
print("logit std", logits.std().item(), "max", logits.max().item())
# per-weight norms
for k in ["wte.weight","wpe.weight"]:
    print(k, sd[k].norm(dim=1).topk(5))
print("ln_f", sd["ln_f.weight"].abs().topk(5))
