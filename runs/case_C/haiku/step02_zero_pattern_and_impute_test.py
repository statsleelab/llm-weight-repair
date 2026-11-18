# (1) Check whether the zeros look like a uniform random 1% mask.
# (2) Validate imputation methods on held-out *nonzero* entries (no answer key used).
import sys
sys.path.insert(0, "/home/user/runs/case_C/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights
from safetensors.torch import load_file
import numpy as np

sd = load_weights("/home/user/runs/case_C/haiku/damaged.safetensors")
torch.manual_seed(0)

def conditional_impute(M, miss, shrink=0.1, passes=2):
    """Gaussian conditional-mean imputation, rows of M are samples, columns are variables.
    M: (R,C) float64 with arbitrary values at missing positions; miss: bool mask (R,C)."""
    X = M.clone()
    obs = ~miss
    R, C = X.shape
    Xo = torch.where(obs, X, torch.zeros_like(X))
    # initial fill: column means of observed
    cnt = obs.sum(0).clamp(min=1)
    colmean = Xo.sum(0) / cnt
    X = torch.where(obs, X, colmean.expand_as(X))
    for _ in range(passes):
        mu = colmean if _ == 0 else X.mean(0)
        Xc = X - mu
        S = Xc.T @ Xc / (R - 1)
        t = S.diagonal().mean()
        S = (1 - shrink) * S + shrink * t * torch.eye(C, dtype=S.dtype)
        P = torch.linalg.inv(S)
        X_new = X.clone()
        rows = torch.nonzero(miss.any(1)).flatten().tolist()
        for i in rows:
            m = torch.nonzero(miss[i]).flatten()
            o = torch.nonzero(obs[i]).flatten()
            xo = X[i, o] - mu[o]
            # E[x_m | x_o] = mu_m - P_mm^{-1} P_mo (x_o - mu_o)
            Pmm = P[m][:, m]
            Pmo = P[m][:, o]
            X_new[i, m] = mu[m] - torch.linalg.solve(Pmm, Pmo @ xo)
        X = X_new
    return X

def stats(name, W):
    Z = (W == 0)
    frac = Z.float().mean().item()
    rowc = Z.sum(1).float()
    colc = Z.sum(0).float()
    print(f"{name:30s} zero_frac={frac:.5f}  per-row zeros mean={rowc.mean():.2f} std={rowc.std():.2f} "
          f"(Poisson std would be {rowc.mean().sqrt():.2f}); per-col std={colc.std():.2f} "
          f"(Poisson {colc.mean().sqrt():.2f}); max row={rowc.max():.0f}")

for k in ["wte.weight", "wpe.weight", "h.0.attn.c_attn.weight", "h.0.attn.c_proj.weight",
          "h.0.mlp.c_fc.weight", "h.0.mlp.c_proj.weight", "h.7.attn.c_attn.weight"]:
    stats(k, sd[k])

# Validation on one representative matrix: hide extra 3% of nonzero entries, impute, measure error there.
for k, transpose in [("h.0.attn.c_attn.weight", True), ("h.0.attn.c_proj.weight", False), ("wpe.weight", False)]:
    W = sd[k].double()
    zero = (W == 0)
    gen = torch.Generator().manual_seed(1)
    test = (~zero) & (torch.rand(W.shape, generator=gen) < 0.03)
    miss = zero | test
    M = W.T.contiguous() if transpose else W
    MM = miss.T.contiguous() if transpose else miss
    TT = test.T.contiguous() if transpose else test
    base_rmse = (M[TT].pow(2).mean().sqrt()).item()  # zero-fill error
    for sh in [0.02, 0.1, 0.3]:
        for passes in [1, 2]:
            Xh = conditional_impute(M, MM, shrink=sh, passes=passes)
            err = (Xh[TT] - M[TT]).pow(2).mean().sqrt().item()
            print(f"{k:28s} shrink={sh:<5} passes={passes} holdout RMSE={err:.5f}  (zero-fill RMSE={base_rmse:.5f}, "
                  f"true RMS={M[TT].pow(2).mean().sqrt().item():.5f})", flush=True)
