import sys
sys.path.insert(0, "/home/user/runs/case_G/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights, Tokenizer, nll_windows
tok = Tokenizer("tokenizer")
ids = tok.encode(open("dev.txt", encoding="utf-8").read())
sd0 = load_weights("damaged.safetensors")
def ppl(sd):
    nll, _ = nll_windows(sd, ids, window=512, max_tokens=4096, batch=4)
    return float(torch.exp(torch.tensor(nll)))
cands = {}
cands["identity"] = sd0
for c in (0.8, 1.0):
    sd = dict(sd0); sd["wte.weight"] = sd0["wte.weight"].clamp(-c, c)
    cands[f"wte_clip_{c}"] = sd
res = {}
for name, sd in cands.items():
    res[name] = ppl(sd)
    print(name, round(res[name], 3), flush=True)
best = min(res, key=res.get)
print("best:", best, flush=True)
if best != "identity":
    save_weights(cands[best], "repaired.safetensors")
    print("saved repaired.safetensors", flush=True)
