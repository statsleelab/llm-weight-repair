import torch, sys, math, time
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
tok = Tokenizer("tokenizer"); ids = tok.encode(open("dev.txt").read())
sd = load_weights("damaged.safetensors")
N = 2048
for t in [6.0]:
    r = repair_all_v2(sd, t, verbose=True)
    nll, acc = nll_windows(r, ids, max_tokens=N); print(f"v2 t={t} ppl {math.exp(nll):.3f} acc {acc:.4f}", flush=True)
    r2 = dict(r); r2["wpe.weight"] = sd["wpe.weight"]
    nll, acc = nll_windows(r2, ids, max_tokens=N); print(f"v2 t={t} wpe-unrepaired ppl {math.exp(nll):.3f} acc {acc:.4f}", flush=True)
    r2 = dict(r); r2["wte.weight"] = sd["wte.weight"]
    nll, acc = nll_windows(r2, ids, max_tokens=N); print(f"v2 t={t} wte-unrepaired ppl {math.exp(nll):.3f} acc {acc:.4f}", flush=True)
    x = torch.tensor([[EOT]+ids[:255]])
    lg, hid = forward(r, x, return_hidden=True)
    print("hidden norms pos>0:", [round(h[0,1:].norm(dim=-1).mean().item(),1) for h in hid])
    print("hidden norms pos0:", [round(h[0,0].norm().item(),1) for h in hid])
    print(generate(r, tok, "The capital of France is", 12))
