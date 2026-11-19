import torch, math
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
tok = Tokenizer("tokenizer"); ids = tok.encode(open("dev.txt").read())
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.ndim==1 and (v.abs().max()>5 or not torch.isfinite(v).all()): print("1d", k, v.abs().max().item(), v.std().item())
r = repair_all(sd, 6.0)
x = torch.tensor([[EOT]+ids[:255]])
for name, m in [("damaged", sd), ("rep", r)]:
    lg, hid = forward(m, x, return_hidden=True)
    print(name, "hidden norms (mean over pos>0):", [round(h[0,1:].norm(dim=-1).mean().item(),1) for h in hid])
    print("  hidden max abs:", [round(h[0,1:].abs().max().item(),1) for h in hid])
    print("  logits absmax", lg.abs().max().item(), "logit std", lg.std().item())
    lp = lg.log_softmax(-1); nll = -lp[0,:-1].gather(-1, x[0,1:,None])[:,0]
    print("  nll mean", nll.mean().item(), "median", nll.median().item(), "top pred", tok.decode(lg[0,-20:].argmax(-1)))
W = r["wte.weight"]; n = W.norm(dim=1)
print("wte row norms: median", n.median().item(), "max", n.max().item(), "top", torch.topk(n,10))
print("wte absmax after repair", W.abs().max().item(), "col absmax top", torch.topk(W.abs().max(0).values,10))
