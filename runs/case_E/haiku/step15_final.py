import sys, math
sys.path.insert(0, "/home/user/runs/case_E/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights, Tokenizer, nll_windows
D = "/home/user/runs/case_E/haiku/"
tok = Tokenizer(D + "tokenizer")
text = open(D + "dev.txt", encoding="utf-8").read()
ids_all = tok.encode(text)
ids = ids_all[:1024]
base = load_weights(D + "damaged.safetensors")
def ev(sd, n_ids):
    nll, acc = nll_windows(sd, n_ids, window=512, max_tokens=len(n_ids))
    return nll, acc
best = None
print("baseline nll(1024)=%.3f" % ev(base, ids)[0], flush=True)
for T in [0.3, 0.4, 0.5, 0.6]:
    sd = dict(base); W = base["wte.weight"].clone(); W[W.abs() > T] = 0.0; sd["wte.weight"] = W
    nll, acc = ev(sd, ids)
    print(f"wte zero |w|>{T}: nll={nll:.3f} top1={acc:.4f}", flush=True)
    if best is None or nll < best[0]:
        best = (nll, T, sd)
nll, T, sd = best
print("chosen T =", T, "nll", round(nll, 3), flush=True)
save_weights(sd, D + "repaired.safetensors")
# full dev evaluation (all ~13.9k tokens) for damaged vs repaired, using nll windows
nd, ad = ev(base, ids_all[:8192]); nr, ar = ev(sd, ids_all[:8192])
print(f"8192-token dev: damaged nll={nd:.3f} ppl={math.exp(nd):.4g} top1={ad:.4f}")
print(f"8192-token dev: repaired nll={nr:.3f} ppl={math.exp(nr):.4g} top1={ar:.4f}")
