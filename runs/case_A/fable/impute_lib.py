"""Ridge / Gaussian-conditional imputation of missing (zeroed) matrix entries.
Pure linear algebra on the weights themselves (no text, no gradients)."""
import torch
torch.set_num_threads(1)

def center(X, miss):
    """double-centre using observed entries only; returns centred X (missing=0) and the additive mean model."""
    obs = (~miss).float()
    Xc = X * obs
    mu = torch.zeros_like(X)
    for _ in range(3):
        r = ((Xc - mu) * obs).sum(1, keepdim=True) / obs.sum(1, keepdim=True).clamp(min=1)
        mu = mu + r
        c = ((Xc - mu) * obs).sum(0, keepdim=True) / obs.sum(0, keepdim=True).clamp(min=1)
        mu = mu + c
    return (X - mu) * obs, mu

def cond_impute(Xc, miss, lam_rel, S=None, V=None):
    """rows = samples. Xc centred with zeros at missing. Returns prediction matrix (only meaningful at miss).
    Gaussian model with covariance C = Xc^T Xc / n + lam*I ; E[x_M | x_O] = -P_MM^{-1} P_MO x_O."""
    n, d = Xc.shape
    if V is None:
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False); V = Vh.T
    ev = S**2 / n
    lam = lam_rel * ev.sum() / d
    # P = V diag(1/(ev+lam) - 1/lam) V^T + I/lam
    coef = 1.0/(ev+lam) - 1.0/lam
    P = (V * coef) @ V.T
    P.diagonal().add_(1.0/lam)
    G = Xc @ P
    cnt = miss.sum(1); K = int(cnt.max())
    out = torch.zeros_like(Xc)
    if K == 0: return out
    # padded index
    idx = torch.full((n, K), d, dtype=torch.long)
    rr, cc = miss.nonzero(as_tuple=True)
    # position within row
    start = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(len(rr)) - start[rr]
    idx[rr, pos] = cc
    Pp = torch.zeros(d+1, d+1); Pp[:d,:d] = P; Pp[d,d] = 1.0
    Gp = torch.cat([G, torch.zeros(n,1)], 1)
    B = 4096
    for s in range(0, n, B):
        ii = idx[s:s+B]
        PMM = Pp[ii[:, :, None], ii[:, None, :]]
        # dummy slots: make identity
        dm = (ii == d)
        PMM = PMM.masked_fill(dm[:, :, None] & dm[:, None, :], 0.0)
        PMM = PMM + torch.diag_embed(dm.float())
        g = Gp[torch.arange(s, min(s+B, n))[:, None], ii]
        z = -torch.linalg.solve(PMM, g.unsqueeze(-1)).squeeze(-1)
        z = z.masked_fill(dm, 0.0)
        tmp = torch.zeros(ii.shape[0], d+1)
        tmp.scatter_(1, ii, z)
        out[s:s+B] = tmp[:, :d]
    return out
