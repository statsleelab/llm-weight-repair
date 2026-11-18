import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, forward, Tokenizer
sd = load_weights("damaged.safetensors")
tok = Tokenizer("tokenizer")
ids = tok.encode(open("dev.txt").read())[:128]
x = torch.tensor([[50256]+ids])
logits, hidden = forward(sd, x, return_hidden=True)
for i,h in enumerate(hidden):
    print("resid", i, "mean norm/token", round(h[0].norm(dim=-1).mean().item(),2), "max", round(h[0].norm(dim=-1).max().item(),1))
print("logit abs max", logits.abs().max().item(), "mean", logits.abs().mean().item())
print("wpe row0 norm", sd["wpe.weight"][0].norm().item(), "row50 norm", sd["wpe.weight"][50].norm().item(), "median row norm", sd["wpe.weight"].norm(dim=1).median().item())
print("wte row norm median", sd["wte.weight"].norm(dim=1).median().item(), "max", sd["wte.weight"].norm(dim=1).max().item())
for i in [0,5,11]:
    print("ln1 w mean", sd[f"h.{i}.ln_1.weight"].mean().item(), "ln2", sd[f"h.{i}.ln_2.weight"].mean().item())
