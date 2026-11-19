import torch, sys, math, time
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
tok = Tokenizer("tokenizer"); ids = tok.encode(open("dev.txt").read())
sd = load_weights("damaged.safetensors")
N = 3072
for kw in eval(sys.argv[1]):
    r, _ = repair_all_v5(sd, verbose=("-v" in sys.argv), **kw)
    nll, acc = nll_windows(r, ids, max_tokens=N); print(f"v5 {kw} ppl {math.exp(nll):.3f} acc {acc:.4f}", flush=True)
x = torch.tensor([[EOT]+ids[:255]]); lg, hid = forward(r, x, return_hidden=True)
print("hidden norms pos0:", [round(h[0,0].norm().item(),1) for h in hid])
