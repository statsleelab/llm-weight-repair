import sys
sys.path.insert(0, "/home/user/runs/case_E/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, forward, EOT
tok = Tokenizer("/home/user/runs/case_E/haiku/tokenizer")
sd = load_weights("/home/user/runs/case_E/haiku/damaged.safetensors")
text = open("/home/user/runs/case_E/haiku/dev.txt", encoding="utf-8").read()
ids = [EOT] + tok.encode(text)[:64]
x = torch.tensor([ids])
logits, hidden = forward(sd, x[:, :-1], return_hidden=True)
print("logits finite:", torch.isfinite(logits).all().item(), "absmax", logits.abs().max().item(), "std", logits.std().item())
for i, h in enumerate(hidden):
    n = h[0].norm(dim=-1)
    print(f"resid {i:2d}: norm mean={n.mean().item():9.2f} max={n.max().item():9.2f} at pos argmax={int(n.argmax())}")
# greedy sample
print(tok.decode(ids[1:20]))
print("pred:", tok.decode(logits[0].argmax(-1).tolist()[:20]))
# embedding row norms
w = sd["wte.weight"]; rn = w.norm(dim=1)
print("wte row norm: median", rn.median().item(), "max", rn.max().item(), "argmax", int(rn.argmax()), "top5", rn.topk(5).indices.tolist(), rn.topk(5).values.tolist())
p = sd["wpe.weight"]; pn = p.norm(dim=1)
print("wpe row norm: first 5", pn[:5].tolist(), "median", pn.median().item(), "max", pn.max().item(), "argmax", int(pn.argmax()))
