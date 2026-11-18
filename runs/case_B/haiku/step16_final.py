import torch, math
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights, nll_windows, Tokenizer
from step07_repair import detect_and_fill
sd0 = load_weights("damaged.safetensors")
tok = Tokenizer("tokenizer"); ids = tok.encode(open("dev.txt").read())
cands = {}
for T in [1.0, 0.7]:
    sd = dict(sd0); sd["wte.weight"], flag = detect_and_fill(sd0["wte.weight"], T)
    nll, acc = nll_windows(sd, ids[:4096])
    print(f"wte T={T} flagged={flag.float().mean().item():.4f} nll4096={nll:.3f}", flush=True)
    cands[T] = (nll, sd)
best = min(cands, key=lambda t: cands[t][0])
print("selected T", best)
sd = cands[best][1]
nll_d, acc_d = nll_windows(sd0, ids)
nll_r, acc_r = nll_windows(sd, ids)
print(f"FULL dev damaged nll={nll_d:.4f} ppl={math.exp(nll_d):.4g} top1={acc_d:.4f}")
print(f"FULL dev repaired nll={nll_r:.4f} ppl={math.exp(nll_r):.4g} top1={acc_r:.4f}")
save_weights(sd, "repaired.safetensors")
from safetensors.torch import load_file
r = load_file("repaired.safetensors"); assert set(r)==set(sd0) and all(r[k].shape==sd0[k].shape for k in sd0)
print("saved and shapes/keys verified", len(r))
