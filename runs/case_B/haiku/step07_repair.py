import sys, math, torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights, Tokenizer, nll_windows, N_LAYER

LINEAR_KEYS = lambda k: k.endswith("weight") and k.split(".")[-2] in ("c_attn", "c_proj", "c_fc") or k in ("wte.weight", "wpe.weight")

def detect_and_fill(w, T, rank=0, iters=0):
    """Flag entries whose deviation from the per-column median exceeds T robust sds
    (scale = matrix-wide MAD). Fill flagged entries with the column median (or a
    low-rank estimate if rank>0). Returns (repaired, flag_mask)."""
    med = w.median(dim=0, keepdim=True).values
    mad = ((w - med).abs().median() * 1.4826).item()
    dev = (w - med).abs() / mad
    flag = dev > T
    fill = med.expand_as(w).clone()
    out = torch.where(flag, fill, w)
    if rank > 0 and iters > 0:
        for _ in range(iters):
            U, S, V = torch.svd_lowrank(out - out.mean(0), q=rank + 10, niter=2)
            L = (U[:, :rank] * S[:rank]) @ V[:, :rank].T + out.mean(0)
            out = torch.where(flag, L, w)
    return out, flag

def repair(sd, T, rank=0, iters=0, skip_wte=False):
    new = {}
    stats = []
    for k, v in sd.items():
        if LINEAR_KEYS(k) and not (skip_wte and k == "wte.weight"):
            r, f = detect_and_fill(v, T, rank, iters)
            new[k] = r
            stats.append((k, f.float().mean().item()))
        else:
            new[k] = v
    return new, stats

if __name__ == "__main__":
    T = float(sys.argv[1]); rank = int(sys.argv[2]); iters = int(sys.argv[3])
    out = sys.argv[4] if len(sys.argv) > 4 else None
    sd = load_weights("damaged.safetensors")
    new, stats = repair(sd, T, rank, iters)
    fr = [s for _, s in stats]
    print(f"T={T} rank={rank} iters={iters} flagged frac mean={sum(fr)/len(fr):.4f} min={min(fr):.4f} max={max(fr):.4f}")
    for k, s in stats:
        if k in ("wte.weight", "wpe.weight", "h.0.mlp.c_fc.weight", "h.11.mlp.c_proj.weight"):
            print("  ", k, round(s, 4))
    tok = Tokenizer("tokenizer")
    ids = tok.encode(open("dev.txt").read())
    nll, acc = nll_windows(new, ids)
    print(f"  dev nll={nll:.4f} ppl={math.exp(nll):.3f} top1={acc:.4f}", flush=True)
    if out:
        save_weights(new, out)
        print("saved", out)
