import torch, math
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
import step18_v6 as V
sd, ids = V.sd, V.ids
A = dict(t=4.2, margin=1.3)
cfg = dict(wte=A, reader=A, writer=dict(t=4.2, margin=1.0, plevel=4.5))
masks = {}
r = V.build(cfg, masks)
ok = all(r[k].shape == sd[k].shape and torch.isfinite(r[k]).all() for k in sd) and set(r) == set(sd)
save_weights(r, "repaired.safetensors")
n = sum(m.sum().item() for m in masks.values()); tot = sum(m.numel() for m in masks.values())
nw = (r["wpe.weight"] != sd["wpe.weight"]).sum().item()
print("keys/shapes ok", ok, "flagged (non-wpe)", n, "frac", n / tot, "wpe changed", nw)
r2 = load_weights("repaired.safetensors")
for N in (8192,):
    nd, ad = nll_windows(sd, ids, max_tokens=N); nr, ar = nll_windows(r2, ids, max_tokens=N)
    print(f"N={N} damaged nll {nd:.3f} ppl {math.exp(min(nd,700)):.3e} acc {ad:.4f} | repaired nll {nr:.4f} ppl {math.exp(nr):.3f} acc {ar:.4f}")
tok = Tokenizer("tokenizer")
print(generate(r2, tok, "The capital of France is", 15))
