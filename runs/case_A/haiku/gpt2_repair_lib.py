"""Helpers for low-rank (hard-impute) completion of zeroed weight entries."""
import torch

def lowrank_proj(X, r):
    """Best rank-r approximation of X (float64), computed via the smaller Gram matrix."""
    m, n = X.shape
    if m >= n:
        G = X.T @ X
        _, V = torch.linalg.eigh(G)
        Vr = V[:, -r:]
        return (X @ Vr) @ Vr.T
    else:
        G = X @ X.T
        _, U = torch.linalg.eigh(G)
        Ur = U[:, -r:]
        return Ur @ (Ur.T @ X)

def hard_impute(W, missing, r, iters=30, tol=1e-7):
    """Fill entries where `missing` is True so that the matrix is approximately rank r.
    Observed entries are kept fixed; missing entries are iteratively set to the rank-r projection."""
    W = W.double()
    obs = ~missing
    X = torch.where(obs, W, torch.zeros_like(W))
    prev = None
    for it in range(iters):
        L = lowrank_proj(X, r)
        Xn = torch.where(obs, W, L)
        if prev is not None:
            delta = (Xn - X)[missing].norm() / max(Xn[missing].norm().item(), 1e-12)
            if delta.item() < tol:
                X = Xn
                break
        prev = True
        X = Xn
    return X.float(), it + 1
