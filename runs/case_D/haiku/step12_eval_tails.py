import torch, sys
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_D/haiku")
from gpt2_lib import load_weights, Tokenizer, perplexity
import numpy as np
tok = Tokenizer("tokenizer")
ids = tok.encode(open("dev.txt", encoding="utf-8").read())
sd = load_weights("damaged.safetensors")
def ppl(s): return perplexity(s, ids, window=512, max_tokens=8000)
names = [k for k, v in sd.items() if v.dim() == 2 and not k.startswith(("wte", "wpe"))]
# robust per-matrix scale from the central part
scale = {}
for k in names:
    a = sd[k].numpy().astype(np.float64).ravel()
    lo, hi = np.quantile(a, [0.25, 0.75])
    scale[k] = (hi - lo) / 1.349
for T in [8.0, 12.0]:
    s2 = dict(sd); tot = 0
    for k in names:
        w = sd[k].clone()
        m = w.abs() > T * scale[k]
        tot += int(m.sum())
        s2[k] = torch.where(m, torch.zeros_like(w), w)
    print(f"T={T}: zeroed {tot} entries ({tot/sum(sd[k].numel() for k in names):.6f} of weights)", flush=True)
    print(f"  ppl={ppl(s2):.3f}", flush=True)
