import sys, math
sys.path.insert(0, "/home/user/runs/case_E/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights, Tokenizer, nll_windows
D = "/home/user/runs/case_E/haiku/"
tok = Tokenizer(D + "tokenizer")
ids = tok.encode(open(D + "dev.txt", encoding="utf-8").read())[:1024]
base = load_weights(D + "damaged.safetensors")
res = []
for T in [0.2, 0.25]:
    sd = dict(base); W = base["wte.weight"].clone(); W[W.abs() > T] = 0.0; sd["wte.weight"] = W
    nll, acc = nll_windows(sd, ids, window=512, max_tokens=1024)
    print(f"wte zero |w|>{T}: nll={nll:.3f} top1={acc:.4f}", flush=True)
    res.append((nll, T, sd))
nll, T, sd = min(res, key=lambda r: r[0])
print("best T", T, flush=True)
save_weights(sd, D + "repaired.safetensors")
