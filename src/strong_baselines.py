"""Strong scripted (non-LLM, gradient-free) repair baselines.

All work per 2-D matrix, oriented so that the 768-wide axis is the feature
axis.  Building blocks:

  ridge_impute   missing entries <- Gaussian conditional mean given the other
                 entries of the same sample (shrunk covariance, CG solve);
                 shrinkage chosen by hiding extra known entries
  em_outliers    two-component scale mixture on locally standardised entries
                 (Student-t bulk + wide Gaussian), EM-fitted; flags entries
                 whose posterior of being noise exceeds 0.5
  block_scan     leave-one-out ridge 'unpredictability' map, box-filter scan
                 for the most anomalous contiguous rectangle

These were written AFTER the agent runs, knowing the four damage families.
"""
from __future__ import annotations
import math
import torch

F = 768
SIDES = [0.03, 0.055, 0.1, 0.17, 0.3, 0.55]   # block side fractions scanned


def orient(w):
    return (w, False) if w.shape[1] == F else (w.T, True)


def fit_prec(X, miss, lam):
    obs = (~miss).float()
    n = obs.sum(0).clamp(min=1)
    mu = (X * obs).sum(0) / n
    Xc = (X - mu) * obs
    S = (Xc.T @ Xc) / (obs.T @ obs).clamp(min=1)
    S = (1 - lam) * S + lam * torch.diag(S.diagonal().clone())
    ev, U = torch.linalg.eigh(S.double())
    ev = ev.clamp(min=1e-3 * ev.mean().clamp(min=1e-12))
    P = ((U / ev) @ U.T).float()
    return mu, Xc, P


def cg_impute(Xc, miss, P, iters=12):
    """Solve P_MM d = -(Xc P)_M for every row at once (masked batched CG)."""
    m = miss.float()
    r = -(Xc @ P) * m
    D = torch.zeros_like(Xc)
    p = r.clone()
    rs = (r * r).sum(1, keepdim=True)
    for _ in range(iters):
        Ap = ((p * m) @ P) * m
        alpha = rs / (p * Ap).sum(1, keepdim=True).clamp(min=1e-30)
        D += alpha * p
        r -= alpha * Ap
        rs_new = (r * r).sum(1, keepdim=True)
        p = r + (rs_new / rs.clamp(min=1e-30)) * p
        rs = rs_new
    return D


def ridge_impute(w, miss, lams=(0.05, 0.3), g=None, max_val_rows=3000):
    """Return w with `miss` entries replaced. Falls back to the feature mean
    when ridge does not beat it on hidden known entries."""
    if not miss.any():
        return w, {"lam": None}
    X, tr = orient(w)
    M = miss.T if tr else miss
    g = g or torch.Generator().manual_seed(0)
    n = X.shape[0]
    idx = torch.randperm(n, generator=g)[:max_val_rows]
    hide = (torch.rand((len(idx), F), generator=g) < 0.03) & ~M[idx]
    best, best_lam, fits = 1.0, None, {}
    for lam in lams:
        Mv = M.clone(); Mv[idx] |= hide
        mu, Xc, P = fit_prec(X, Mv, lam)
        d = cg_impute(Xc[idx], Mv[idx], P)
        truth = (X[idx] - mu)[hide]
        ratio = ((d[hide] - truth) ** 2).sum().item() / (truth ** 2).sum().clamp(min=1e-30).item()
        if math.isfinite(ratio) and ratio < best:
            best, best_lam = ratio, lam
    mu, Xc, P = fit_prec(X, M, best_lam if best_lam is not None else 0.3)
    out = torch.where(M, mu.expand_as(X), X)
    if best_lam is not None and best < 0.98:
        d = cg_impute(Xc, M, P)
        if torch.isfinite(d).all():
            out = torch.where(M, mu + d, X)
    return (out.T if tr else out).contiguous(), {"lam": best_lam, "val_ratio": best}


def colmean_fill(w, miss):
    X, tr = orient(w)
    M = miss.T if tr else miss
    obs = (~M).float()
    mu = (X * obs).sum(0) / obs.sum(0).clamp(min=1)
    out = torch.where(M, mu.expand_as(X), X)
    return (out.T if tr else out).contiguous()


def em_outliers(w, miss, g=None, iters=40, nu=5.0):
    X, tr = orient(w)
    M = miss.T if tr else miss
    Xn = X.masked_fill(M, float("nan"))
    med = Xn.nanmedian(0).values
    R = Xn - med
    a = (R.abs().nanmedian(1, keepdim=True).values / 0.6745).clamp(min=1e-8)
    b = ((R.abs() / a).nanmedian(0).values / 0.6745).clamp(min=1e-8)
    u = (R / (a * b)).nan_to_num(0.0)
    g = g or torch.Generator().manual_seed(0)
    flat = u.flatten()
    s = flat[torch.randint(0, flat.numel(), (1_000_000,), generator=g)].double()
    c = 0.6745 / {5.0: 0.7267, 30.0: 0.6828}[nu]
    def f0(x):
        t = x / c
        return torch.exp(torch.lgamma(torch.tensor((nu + 1) / 2)) - torch.lgamma(torch.tensor(nu / 2))) / math.sqrt(nu * math.pi) * (1 + t * t / nu) ** (-(nu + 1) / 2) / c
    def f1(x, tau):
        return torch.exp(-0.5 * (x / tau) ** 2) / (tau * math.sqrt(2 * math.pi))
    pi, tau = 0.01, 8.0
    for _ in range(iters):
        p1 = pi * f1(s, tau); post = p1 / (p1 + (1 - pi) * f0(s))
        pi = float(post.mean().clamp(1e-6, 0.6)); tau = float(((post * s * s).sum() / post.sum().clamp(min=1e-12)).sqrt())
    info = {"pi": pi, "tau": tau}
    if pi < 2e-4 or tau < 2.5:
        flag = torch.zeros_like(M)
    else:
        ud = u.double()
        p1 = pi * f1(ud, tau)
        flag = (p1 / (p1 + (1 - pi) * f0(ud)) > 0.5) & ~M
    return (flag.T if tr else flag), info


def block_scan(w, miss, lam=0.1, zthr=7.5):
    X, tr = orient(w)
    M = miss.T if tr else miss
    mu, Xc, P = fit_prec(X, M, lam)
    q = ((Xc @ P) ** 2 / P.diagonal()).clamp(max=16.0)
    q[M] = q[~M].median()
    q = ((q - q.mean()) / q.std()).double()
    n = q.shape[0]
    I = torch.zeros(n + 1, F + 1, dtype=torch.float64)
    I[1:, 1:] = q.cumsum(0).cumsum(1)
    best, best_raw = (-1e9, None), -1e9
    for sh in SIDES:
        for sw in SIDES:
            h, wd = max(2, round(n * sh)), max(2, round(F * sw))
            S = (I[h:, wd:] - I[:-h, wd:] - I[h:, :-wd] + I[:-h, :-wd]) / math.sqrt(h * wd)
            med = S.median(); sd = (S - med).abs().median() * 1.4826 + 1e-12
            k = int(S.argmax()); z = float((S.flatten()[k] - med) / sd)
            raw = float(S.flatten()[k])
            if raw > best_raw:
                best_raw, best = raw, (z, (k // S.shape[1], k % S.shape[1], h, wd))
    flag = torch.zeros_like(M)
    z, rect = best
    if z > zthr:
        r0, c0, h, wd = rect
        flag[r0:r0 + h, c0:c0 + wd] = True
        flag &= ~M
    return (flag.T if tr else flag), {"z": z, "rect": rect if z > zthr else None}


def repair_all(sd, mode, masks=None, seed=0):
    """mode: colmean_zero | zero_ridge | em_ridge | block_ridge | oracle_colmean | oracle_ridge"""
    g = torch.Generator().manual_seed(seed)
    out, info = {}, {}
    for k, w in sd.items():
        if w.ndim != 2:
            out[k] = w; continue
        zero = w == 0
        if mode == "colmean_zero":
            out[k] = colmean_fill(w, zero); continue
        if mode == "oracle_colmean":
            out[k] = colmean_fill(w, masks[k]); continue
        if mode == "oracle_ridge":
            out[k], info[k] = ridge_impute(w, masks[k], g=g); continue
        miss = zero
        if mode in ("em_ridge", "emg_ridge"):
            fl, inf = em_outliers(w, zero, g=g, nu=5.0 if mode == "em_ridge" else 30.0); miss = zero | fl
        elif mode == "block_ridge":
            w0, _ = ridge_impute(w, zero, g=g) if zero.any() else (w, None)
            fl, inf = block_scan(w0, zero); miss = zero | fl
        else:
            inf = {}
        out[k], inf2 = ridge_impute(w, miss, g=g)
        info[k] = inf | inf2 | {"n_flag": int(miss.sum())}
    return out, info
