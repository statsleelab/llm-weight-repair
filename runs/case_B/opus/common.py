import torch
def scale_model(w, iters=6, mask=None):
    """sigma_ij = r_i * c_j, robust rank-1 fit of |w| (median-based), ignoring mask (True=excluded)."""
    A = w.abs().clone()
    if mask is not None: A[mask] = float('nan')
    r = torch.nanmedian(A, 1).values.clamp_min(1e-8)
    c = torch.ones(w.shape[1])
    for _ in range(iters):
        c = torch.nanmedian(A / r[:, None], 0).values.clamp_min(1e-8)
        r = torch.nanmedian(A / c[None, :], 1).values.clamp_min(1e-8)
    return r[:, None] * c[None, :] * 1.4826
def zscore(w, iters=6, rounds=2, T=8.0):
    mask=None
    for _ in range(rounds):
        s = scale_model(w, iters, mask)
        z = w / s
        mask = z.abs() > T
    return z, s

def repair_generic(w, T, fill="zero"):
    z, s = zscore(w)
    m = z.abs() > T
    out = w.clone(); out[m] = 0.0
    return out, m

def repair_wpe(w, T=6.0):
    # positional embeddings are smooth along position: predict each entry from neighbours
    W = w.clone(); n = W.shape[0]
    allm = torch.zeros_like(W, dtype=torch.bool)
    for _ in range(3):
        P = torch.full((n, 4, W.shape[1]), float('nan'))
        for j, d in enumerate([-2, -1, 1, 2]):
            idx = torch.arange(n) + d; ok = (idx >= 0) & (idx < n)
            P[ok, j] = W[idx[ok]]
        pred = torch.nanmedian(P, 1).values
        r = W - pred
        sc = r.abs().median(0, keepdim=True).values * 1.4826 + 1e-6
        m = (r / sc).abs() > T
        # don't let first rows (genuinely irregular) be over-flagged: require also global outlier there
        if not m.any(): break
        W[m] = pred[m]; allm |= m
    return W, allm

def repair_protect(w, T, protect_k=5.0):
    z, s = zscore(w)
    m = z.abs() > T
    tau = w[z.abs() > 15].std().item() if (z.abs() > 15).sum() > 50 else 1.0
    m &= w.abs() < protect_k * tau
    out = w.clone(); out[m] = 0.0
    return out, m, tau

def location_model(w, mask=None, iters=4):
    """additive robust fit mu_ij = a_i + b_j (median polish), ignoring masked entries."""
    R = w.clone()
    if mask is not None: R[mask] = float('nan')
    a = torch.zeros(w.shape[0]); b = torch.zeros(w.shape[1])
    for _ in range(iters):
        db = torch.nanmedian(R, 0).values; b += db; R -= db[None, :]
        da = torch.nanmedian(R, 1).values; a += da; R -= da[:, None]
    return a[:, None] + b[None, :]

def detect2(w, T=6.0, rounds=3, Tmask=6.0):
    mask = None
    for _ in range(rounds):
        mu = location_model(w, mask)
        r = w - mu
        s = scale_model(r, 6, mask).clamp_min(1e-6)
        z = r / s
        mask = z.abs() > Tmask
    return z, mu, s

def repair2(w, T, protect_k=5.0):
    z, mu, s = detect2(w)
    r = w - mu
    big = z.abs() > 15
    tau = r[big].std().item() if big.sum() > 50 else 1.0
    m = (z.abs() > T) & (r.abs() < protect_k * tau)
    out = w.clone(); out[m] = mu[m]
    return out, m, tau

def detect3(w, c=2.6, q=0.95):
    """flag entries outlying in BOTH their row and their column, measured against the
    row's / column's own tail scale (q-quantile of |residual|), after median-polish centering."""
    z, mu, s = detect2(w)
    r = w - mu
    A = r.abs()
    def qn(x, dim):
        # quantile along dim (use kthvalue for memory)
        n = x.shape[dim]; k = max(1, int(round(q * n)))
        return x.kthvalue(k, dim=dim, keepdim=True).values.clamp_min(1e-7)
    rq = qn(A, 1); cq = qn(A, 0)
    m = (A > c * rq) & (A > c * cq)
    return m, mu, r

def repair3(w, c=2.6, protect_k=5.0):
    m, mu, r = detect3(w, c)
    big = m
    tau = r[big].std().item()
    m = m & (r.abs() < protect_k * tau)
    out = w.clone(); out[m] = mu[m]
    return out, m, tau

def repair_wpe2(w, T=6.0, edge=8, c=2.6):
    W, m = repair_wpe(w, T)
    # edge rows are genuinely irregular -> restore and use generic row/col test instead
    W[:edge] = w[:edge]; W[-2:] = w[-2:]; m[:edge] = False; m[-2:] = False
    m2, mu, r = detect3(w, c)
    for rows in [slice(0, edge), slice(w.shape[0]-2, w.shape[0])]:
        mm = m2[rows]
        W[rows][mm] = 0  # placeholder replaced below
        Wr = w[rows].clone(); Wr[mm] = mu[rows][mm]; W[rows] = Wr; m[rows] = mm
    return W, m

def estimate_tau(r, s):
    sel = (r.abs() > 15 * s) & (15 * s < 0.15)
    if sel.sum() < 30: sel = r.abs() > 15 * s
    return (r[sel].abs().median() / 0.6745).item(), int(sel.sum())

def detect4(w, q=0.98, iters=2):
    """detect2 + per-row / per-column tail calibration: rows/cols whose clean tail is
    genuinely heavier than the rank-1 scale model predicts get their scale inflated."""
    z, mu, s = detect2(w)
    r = w - mu
    f = torch.ones(w.shape[0], 1); g = torch.ones(1, w.shape[1])
    for _ in range(iters):
        za = (r / (s * f * g)).abs()
        tr = za.quantile(q, dim=1, keepdim=True) if za.shape[1] < 16000000 else None
        tc = torch.stack([c.quantile(q) for c in za.split(4096, 1)]) if False else None
        tr = torch.quantile(za, q, dim=1, keepdim=True)
        f = f * (tr / tr.median()).clamp_min(1.0)
        za = (r / (s * f * g)).abs()
        n = za.shape[0]; k = int(q * n)
        tc = za.kthvalue(k, dim=0, keepdim=True).values
        g = g * (tc / tc.median()).clamp_min(1.0)
    sc = s * f * g
    return r / sc, mu, sc

def repair4(w, T, pk=3.5, q=0.98):
    z, mu, sc = detect4(w, q)
    r = w - mu
    tau, n = estimate_tau(r, sc)
    m = (z.abs() > T) & (r.abs() < pk * tau)
    out = w.clone(); out[m] = mu[m]
    return out, m, tau

import math as _m
def soft_repair(w, z, mu, s, nu=4.0, pi=0.01, tau=None, hard=False):
    """posterior-mean estimate under mixture: clean ~ mu + s*t_nu, corrupt ~ N(0,tau^2) (replacement)."""
    r = w - mu
    if tau is None:
        tot = (w**2).mean(); clean = (r[z.abs() < 5]**2).mean()
        tau = ((tot - clean) / pi).clamp_min(1e-4).sqrt().item()
    # log densities
    lc = (torch.lgamma(torch.tensor((nu+1)/2)) - torch.lgamma(torch.tensor(nu/2)) - 0.5*_m.log(nu*_m.pi)) \
         - torch.log(s) - (nu+1)/2*torch.log1p((r/s)**2/nu)
    lk = -0.5*_m.log(2*_m.pi) - _m.log(tau) - 0.5*(w/tau)**2
    logit = _m.log(pi) + lk - (_m.log(1-pi) + lc)
    p = torch.sigmoid(logit)
    if hard: p = (p > 0.5).float()
    return mu + (1 - p) * r, p, tau
