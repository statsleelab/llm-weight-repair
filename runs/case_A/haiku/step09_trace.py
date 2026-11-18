import sys, math
import torch
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_A/haiku")
from gpt2_lib import load_weights, forward, Tokenizer, EOT
tok = Tokenizer("/home/user/runs/case_A/haiku/tokenizer")
ids = tok.encode(open("/home/user/runs/case_A/haiku/dev.txt", encoding="utf-8").read())[:256]
path = sys.argv[1]
sd = load_weights(path)
x = torch.tensor([[EOT] + ids[:127]])
logits, hidden = forward(sd, x, return_hidden=True)
for i, h in enumerate(hidden):
    print(f"resid after block {i:2d}: rms={h.pow(2).mean().sqrt().item():10.3f} max|.|={h.abs().max().item():12.2f}", flush=True)
print("logits max|.|", logits.abs().max().item(), "ln_f-free check done")
print("ln_f.weight abs max", sd["ln_f.weight"].abs().max().item(), "wte abs max", sd["wte.weight"].abs().max().item())
