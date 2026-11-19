import sys, time
sys.path.insert(0, "/home/user/runs/case_E/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, nll_windows
tok = Tokenizer("/home/user/runs/case_E/haiku/tokenizer")
ids = tok.encode(open("/home/user/runs/case_E/haiku/dev.txt", encoding="utf-8").read())[:1024]
base = load_weights("/home/user/runs/case_E/haiku/damaged.safetensors")
W = base["wte.weight"]
def ev(sd):
    nll, acc = nll_windows(sd, ids, window=512, max_tokens=1024)
    return nll, acc
def hard_impute(W, M, rank, iters):
    X = W.clone(); X[M] = 0.0
    for it in range(iters):
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        R = (U[:, :rank] * S[:rank]) @ Vh[:rank]
        X = torch.where(M, R, W)
    return X
# spectrum of the wte (with big entries clipped to the bulk) for reference
Ssv = torch.linalg.svdvals(W.clamp(-0.5, 0.5))
e = (Ssv**2).cumsum(0) / (Ssv**2).sum()
print("energy in top 64/128/256/384 dims:", [round(float(e[r-1]),3) for r in [64,128,256,384]], flush=True)
# robust per-matrix sigma
g = torch.Generator().manual_seed(0)
s = W.flatten()[torch.randperm(W.numel(), generator=g)[:300000]]
sig = ((torch.quantile(s, 0.75) - torch.quantile(s, 0.25)) / 1.349).item()
print("wte robust sigma", sig, flush=True)
for T_mult in [4.0]:
    M = W.abs() > T_mult * sig
    print(f"threshold {T_mult}sig={T_mult*sig:.3f} flagged={int(M.sum())} ({M.float().mean().item()*100:.3f}%)", flush=True)
    for rank in [128, 384]:
        t = time.time()
        X = hard_impute(W, M, rank, iters=12)
        sd = dict(base); sd["wte.weight"] = X
        nll, acc = ev(sd)
        print(f"  rank={rank} nll={nll:.4f} top1={acc:.4f} ({time.time()-t:.1f}s)", flush=True)
        torch.save(X, f"/home/user/runs/case_E/haiku/wte_fix_r{rank}.pt")
