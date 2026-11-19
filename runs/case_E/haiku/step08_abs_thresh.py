import sys, time
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
def apply(T, mode, include_emb):
    out = dict(base)
    n = 0
    for k, W in base.items():
        if W.dim() != 2: continue
        if ("wte" in k or "wpe" in k) and not include_emb: continue
        if "c_attn" not in k and "c_proj" not in k and "c_fc" not in k and "wte" not in k and "wpe" not in k: continue
        m = W.abs() > T
        n += int(m.sum())
        W2 = W.clone()
        if mode == "zero": W2[m] = 0.0
        else: W2 = torch.clamp(W2, -T, T)
        out[k] = W2
    return out, n
for T in [2.0, 3.0, 4.0, 5.0]:
    for mode in ["zero", "clip"]:
        sd, n = apply(T, mode, include_emb=False)
        p, a = ppl(sd)
        print(f"T={T} mode={mode:5s} n_changed={n:6d} ppl={p:12.3f} top1={a:.4f}", flush=True)
