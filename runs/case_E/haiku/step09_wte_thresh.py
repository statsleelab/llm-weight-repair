import sys
sys.path.insert(0, "/home/user/runs/case_E/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, nll_windows
tok = Tokenizer("/home/user/runs/case_E/haiku/tokenizer")
ids = tok.encode(open("/home/user/runs/case_E/haiku/dev.txt", encoding="utf-8").read())[:2048]
base = load_weights("/home/user/runs/case_E/haiku/damaged.safetensors")
def ppl(sd):
    nll, acc = nll_windows(sd, ids, window=512, max_tokens=2048)
    return torch.exp(torch.tensor(nll)).item(), acc
W = base["wte.weight"]
print("wte quantiles |w|:", [round(float(q),3) for q in torch.quantile(W.abs().flatten()[torch.randperm(W.numel(), generator=torch.Generator().manual_seed(1))[:500000]], torch.tensor([0.5,0.9,0.99,0.999]))])
print("wte std of rows-median-abs", W.abs().median(dim=1).values.mean().item())
for T in [0.5, 0.8, 1.0, 1.5]:
    sd = dict(base); W2 = W.clone(); m = W2.abs() > T; W2[m] = 0.0; sd["wte.weight"] = W2
    p, a = ppl(sd); print(f"wte zero |w|>{T}: n={int(m.sum())} ppl={p:.3f} top1={a:.4f}", flush=True)
# wte with all rows' outliers replaced by tied-head-safe: check also the LM head via embed inputs only
