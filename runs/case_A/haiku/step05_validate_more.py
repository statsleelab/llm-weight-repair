import sys, time
import torch
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_A/haiku")
from gpt2_repair_lib import hard_impute
from safetensors.torch import load_file
sd = load_file("/home/user/runs/case_A/haiku/damaged.safetensors")
g = torch.Generator().manual_seed(1)
cases = [
    ("h.0.attn.c_proj.weight", None, [8, 16, 32, 64, 96]),
    ("h.0.mlp.c_fc.weight", None, [8, 16, 32, 64, 128]),
    ("wte.weight", 12000, [8, 16, 32, 64, 128, 256]),
    ("wpe.weight", None, [4, 8, 12, 16, 24]),
]
for name, nrow, ranks in cases:
    W = sd[name].float()
    if nrow: W = W[:nrow]
    zero = (W == 0)
    H = (torch.rand(W.shape, generator=g) < 0.01) & (~zero)
    missing = zero | H
    rms_true = W[H].pow(2).mean().sqrt().item()
    print(f"== {name} {tuple(W.shape)} hidden={int(H.sum())} zero-fill RMSE={rms_true:.4f}", flush=True)
    for r in ranks:
        t0 = time.time()
        X, its = hard_impute(W, missing, r, iters=40)
        err = (X[H] - W[H]).pow(2).mean().sqrt().item()
        print(f"   rank {r:4d}: ratio={err/rms_true:.3f} iters={its} t={time.time()-t0:.1f}s", flush=True)
