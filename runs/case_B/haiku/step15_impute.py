import torch, math
torch.set_num_threads(1)
from gpt2_lib import load_weights, nll_windows, Tokenizer
from step07_repair import detect_and_fill
sd0 = load_weights("damaged.safetensors")
tok = Tokenizer("tokenizer"); ids = tok.encode(open("dev.txt").read())
def wte_fix(T, rank, iters):
    return detect_and_fill(sd0["wte.weight"], T, rank, iters)[0]
def wpe_abs(W, thr):
    out = W.clone(); out[W.abs()>thr] = 0.0
    return out
res = {}
for name,(T,r,it) in {"wteT2_r128":(2,128,5), "wteT1_r64":(1,64,5), "wteT2_r32":(2,32,5)}.items():
    sd = dict(sd0); sd["wte.weight"] = wte_fix(T,r,it)
    sd["wpe.weight"] = wpe_abs(sd0["wpe.weight"], 0.1)
    nll, acc = nll_windows(sd, ids[:4096])
    print(f"{name:12s} nll4096={nll:.3f} ppl={math.exp(nll):.2f} top1={acc:.4f}", flush=True)
sd = dict(sd0); sd["wte.weight"] = detect_and_fill(sd0["wte.weight"],1.0)[0]
sd["wpe.weight"] = wpe_abs(sd0["wpe.weight"], 0.1)
nll, acc = nll_windows(sd, ids[:4096]); print(f"wteT1 + wpe_abs0.1 nll={nll:.3f} ppl={math.exp(nll):.2f}")
