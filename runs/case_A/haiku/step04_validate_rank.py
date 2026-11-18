"""Validation of the completion method: hide an extra random 1% of the *observed* entries,
impute them together with the true zeros, and measure error on the hidden entries only."""
import sys, time
import torch
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_A/haiku")
from gpt2_repair_lib import hard_impute
from safetensors.torch import load_file

sd = load_file("/home/user/runs/case_A/haiku/damaged.safetensors")
g = torch.Generator().manual_seed(0)
tests = {
    "h.0.attn.c_attn.weight": None,
    "h.6.mlp.c_proj.weight": None,
    "wpe.weight": None,
}
ranks = [8, 16, 32, 64, 128, 256]
for name in tests:
    W = sd[name].float()
    zero = (W == 0)
    H = (torch.rand(W.shape, generator=g) < 0.01) & (~zero)   # hidden observed entries
    missing = zero | H
    rms_true = W[H].pow(2).mean().sqrt().item()
    print(f"== {name} {tuple(W.shape)}  hidden={int(H.sum())}  zero-fill RMSE on hidden={rms_true:.4f}")
    for r in ranks:
        t0 = time.time()
        X, its = hard_impute(W, missing, r, iters=40)
        err = (X[H] - W[H]).pow(2).mean().sqrt().item()
        print(f"   rank {r:4d}: RMSE={err:.4f} (ratio {err/rms_true:.3f}) iters={its} t={time.time()-t0:.1f}s", flush=True)
