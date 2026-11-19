import torch, sys, math, time
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
tok = Tokenizer("tokenizer"); ids = tok.encode(open("dev.txt").read())
sd = load_weights("damaged.safetensors")
N = 3072
def fam(k):
    if k == "wte.weight": return "wte"
    if k.endswith("c_proj.weight"): return "writer"
    return "reader"
cache = {}
def rep(k, **kw):
    key = (k, tuple(sorted(kw.items())))
    if key not in cache:
        v = sd[k]; m, cm, _ = detect_v5(v, **kw); R = v.clone(); R[m] = cm[m]; cache[key] = R
    return cache[key]
wpe = repair_wpe(sd["wpe.weight"], 6.0)[0]
def build(cfg):
    out = {}
    for k, v in sd.items():
        if v.ndim != 2: out[k] = v
        elif k == "wpe.weight": out[k] = wpe
        else: out[k] = rep(k, **cfg[fam(k)])
    return out
if __name__ == "__main__":
    base = dict(t=4.2, margin=1.3)
    for name, alt in eval(sys.argv[1]):
        cfg = {f: dict(base) for f in ("wte", "writer", "reader")}
        cfg[name] = dict(base, **alt)
        nll, acc = nll_windows(build(cfg), ids, max_tokens=N); print(f"family {name} {alt} ppl {math.exp(nll):.3f} acc {acc:.4f}", flush=True)
