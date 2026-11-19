# Repair 2-D weight matrices: flag entries far outside the robust core of the
# matrix's own distribution, then impute flagged entries by a low-rank
# (hard-impute) estimate computed from the unflagged entries.
# usage: python step09_repair2d.py THR RANK KEY [KEY ...]   (RANK 0 = zero-impute)
import torch, sys, time
torch.set_num_threads(1)
from safetensors.torch import load_file
THR = float(sys.argv[1]); R = int(sys.argv[2]); keys = sys.argv[3:]
sd = load_file("damaged.safetensors")
def core_std(x):
    q = torch.quantile(x[::3], torch.tensor([.25,.75]))
    return ((q[1]-q[0])/1.349).item()
for k in keys:
    t0 = time.time()
    W = sd[k].float()
    s0 = core_std(W.flatten()); med = W.median().item()
    mask = (W - med).abs() > THR * s0
    X = W.clone(); X[mask] = 0.0
    if R == 0:
        Wf = X
    else:
        Y = X.clone()
        for it in range(4):
            U, S, V = torch.svd_lowrank(Y, q=R + 32, niter=2)
            L = (U[:, :R] * S[:R]) @ V[:, :R].T
            Y[mask] = L[mask]
        Wf = torch.where(mask, L, W)
    torch.save({"W": Wf, "mask": mask}, f"cache/{k}__t{THR}_r{R}.pt")
    print(f"{k} s0={s0:.4f} flagged={mask.float().mean().item()*100:.2f}% t={time.time()-t0:.1f}s", flush=True)
