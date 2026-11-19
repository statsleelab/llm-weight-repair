import torch, numpy as np
torch.set_num_threads(1)

def rsig(x, dim):
    med = x.median(dim=dim, keepdim=True).values
    mad = (x - med).abs().median(dim=dim, keepdim=True).values
    return med, 1.4826 * mad

def detect_generic(W, t):
    """flag entries far from both their row and column robust distribution."""
    g = 1.4826 * W.abs().median()
    rm, rs = rsig(W, 1)
    cm, cs = rsig(W, 0)
    # rank-1 scale model, never smaller than the global scale
    s = torch.maximum(rs, cs)
    s = torch.maximum(s, g.expand_as(s))
    z = torch.minimum((W - rm).abs(), (W - cm).abs()) / s
    return z > t, cm.expand_as(W)

def repair_generic(W, t, fill="colmed"):
    m, cm = detect_generic(W, t)
    R = W.clone()
    R[m] = cm[m] if fill == "colmed" else 0.0
    return R, m

def medfilt_pos(W, k):
    # median along dim 0 over window k (excluding nothing), reflect padding
    h = k // 2
    idx = torch.arange(W.shape[0])
    stack = []
    for d in range(-h, h + 1):
        if d == 0: continue
        j = (idx + d)
        j = torch.where(j < 0, idx + (h + 1 - d if False else -d + 0), j)  # placeholder, fixed below
        stack.append(j)
    # simpler: build neighbour index lists with clamping to valid range by mirroring to the other side
    stack = []
    for d in range(1, h + 1):
        for sgn in (-1, 1):
            j = idx + sgn * d
            bad = (j < 0) | (j >= W.shape[0])
            j = torch.where(bad, idx - sgn * (d + h), j)  # take further neighbour on valid side
            j = j.clamp(0, W.shape[0] - 1)
            stack.append(W[j])
    return torch.stack(stack, 0).median(0).values

def repair_wpe(W, t, k=6, iters=2):
    R = W.clone(); M = torch.zeros_like(W, dtype=torch.bool)
    for _ in range(iters):
        mf = medfilt_pos(R, k)
        res = R - mf
        # scale: per-column robust sigma of the residual, plus per-row floor
        _, cs = rsig(res, 0)
        _, rs = rsig(res, 1)
        s = torch.maximum(cs, rs) + 1e-4
        m = res.abs() > t * s
        m[0] = False                      # position 0 is genuinely unlike its neighbours
        R[m] = mf[m]; M |= m
    # row 0: only its own-row statistics are available
    med, s0 = rsig(W[0:1], 1)
    m0 = ((W[0:1] - med).abs() > 12 * s0)[0]
    return R, M, m0

def repair_all(sd, t, t_wpe=6.0, fill="colmed", verbose=False):
    out = {}; nflag = 0; ntot = 0
    for k, v in sd.items():
        if v.ndim != 2: out[k] = v; continue
        if k == "wpe.weight":
            R, m, m0 = repair_wpe(v, t_wpe)
            if verbose: print(k, "flagged", int(m.sum()), "row0 suspicious", int(m0.sum()))
        else:
            R, m = repair_generic(v, t, fill)
            if verbose: print(f"{k:28s} flagged {int(m.sum()):7d} frac {m.float().mean().item():.5f}")
        nflag += int(m.sum()); ntot += v.numel(); out[k] = R
    if verbose: print("total flagged frac", nflag / ntot)
    return out

# ---------------- v2: line-count-capped scales -----------------
def line_scale(D, dim, t, vis_rate=0.00055):
    """D = |deviation|. Robust scale per line along `dim`, raised so that no line has more
    flags than uniform-random corruption could plausibly put there."""
    n = D.shape[dim]
    lam = vis_rate * n
    K = int(lam + 4 * lam ** 0.5 + 3)
    kth = torch.topk(D, K + 1, dim=dim).values.select(dim, K).unsqueeze(dim)
    mad = 1.4826 * D.median(dim=dim, keepdim=True).values
    return torch.maximum(mad, kth / t)

def detect_v2(W, t):
    g = 1.4826 * W.abs().median()
    rm = W.median(dim=1, keepdim=True).values
    cm = W.median(dim=0, keepdim=True).values
    Dr = (W - rm).abs(); Dc = (W - cm).abs()
    s = torch.maximum(line_scale(Dr, 1, t), line_scale(Dc, 0, t))
    s = torch.maximum(s, g.expand_as(s))
    z = torch.minimum(Dr, Dc) / s
    return z > t, cm.expand_as(W)

def repair_all_v2(sd, t, t_wpe=6.0, fix_wpe=True, verbose=False, skip=()):
    out = {}; nflag = 0; ntot = 0
    for k, v in sd.items():
        if v.ndim != 2 or any(s in k for s in skip): out[k] = v; continue
        if k == "wpe.weight":
            if not fix_wpe: out[k] = v; continue
            R, m, m0 = repair_wpe(v, t_wpe)
        else:
            m, cm = detect_v2(v, t); R = v.clone(); R[m] = cm[m]
        if verbose: print(f"{k:28s} flagged {int(m.sum()):7d} frac {m.float().mean().item():.5f}")
        nflag += int(m.sum()); ntot += v.numel(); out[k] = R
    if verbose: print("total flagged frac", nflag / ntot)
    return out

# ---------------- v3: multiplicative (rank-1) local scale + count caps -----------------
def cap_mult(Z, dim, t, vis_rate=0.001):
    n = Z.shape[dim]; lam = vis_rate * n
    K = int(lam + 4 * lam ** 0.5 + 3)
    kth = torch.topk(Z, K + 1, dim=dim).values.select(dim, K).unsqueeze(dim)
    return (kth / t).clamp(min=1.0)

def detect_v3(W, t):
    cm = W.median(dim=0, keepdim=True).values
    D = (W - cm).abs()
    c = 1.4826 * D.median(dim=0, keepdim=True).values + 1e-8
    Z = D / c
    r = 1.4826 * Z.median(dim=1, keepdim=True).values + 1e-8
    Z = Z / r
    Z = Z / torch.maximum(cap_mult(Z, 1, t), cap_mult(Z, 0, t))
    return Z > t, cm.expand_as(W)

def repair_all_v3(sd, t, t_wpe=6.0, verbose=False):
    out = {}; nflag = 0; ntot = 0
    for k, v in sd.items():
        if v.ndim != 2: out[k] = v; continue
        if k == "wpe.weight":
            R, m, m0 = repair_wpe(v, t_wpe)
        else:
            m, cm = detect_v3(v, t); R = v.clone(); R[m] = cm[m]
        if verbose: print(f"{k:28s} flagged {int(m.sum()):7d} frac {m.float().mean().item():.5f}")
        nflag += int(m.sum()); ntot += v.numel(); out[k] = R
    if verbose: print("total flagged frac", nflag / ntot)
    return out

# ---------------- v4: asymmetric quantile scales per column, row factor, caps, giant protection ---------
def kq(X, q, dim):
    n = X.shape[dim]; k = min(n, max(1, int(round(q * (n - 1))) + 1))
    return torch.kthvalue(X, k, dim=dim, keepdim=True).values

def detect_v4(W, t, protect=4.2):
    cm = W.median(dim=0, keepdim=True).values
    D = W - cm
    su = kq(D, 0.97, 0) / 1.881 + 1e-9
    sl = -kq(D, 0.03, 0) / 1.881 + 1e-9
    Z = torch.where(D > 0, D / su, -D / sl)
    r = kq(Z, 0.97, 1) / 2.17 + 1e-9
    Z = Z / r
    Z = Z / torch.maximum(cap_mult(Z, 1, t), cap_mult(Z, 0, t))
    noise_sd = 10.0 * W.std() / 1.0488
    return (Z > t) & (D.abs() < protect * noise_sd), cm.expand_as(W)

def repair_all_v4(sd, t, t_wpe=6.0, protect=4.2, verbose=False):
    out = {}; nflag = 0; ntot = 0
    for k, v in sd.items():
        if v.ndim != 2: out[k] = v; continue
        if k == "wpe.weight":
            R, m, m0 = repair_wpe(v, t_wpe)
        else:
            m, cm = detect_v4(v, t, protect); R = v.clone(); R[m] = cm[m]
        if verbose: print(f"{k:28s} flagged {int(m.sum()):7d} frac {m.float().mean().item():.5f}")
        nflag += int(m.sum()); ntot += v.numel(); out[k] = R
    if verbose: print("total flagged frac", nflag / ntot)
    return out

# ---------------- v5: v4 + margin over the K-th value in crowded lines + structural giant protection ----
def cap_mult2(Z, dim, t, margin, rate=0.001):
    n = Z.shape[dim]; lam = rate * n
    K = int(lam + 4 * lam ** 0.5 + 3)
    kth = torch.topk(Z, K + 1, dim=dim).values.select(dim, K).unsqueeze(dim)
    return (margin * kth / t).clamp(min=1.0)

def detect_v5(W, t, margin=1.3, protect=4.2, hot_lvl=2.5):
    cm = W.median(dim=0, keepdim=True).values
    D = W - cm
    su = kq(D, 0.97, 0) / 1.881 + 1e-9
    sl = -kq(D, 0.03, 0) / 1.881 + 1e-9
    Z = torch.where(D > 0, D / su, -D / sl)
    r = kq(Z, 0.97, 1) / 2.17 + 1e-9
    Z = Z / r
    Z = Z / torch.maximum(cap_mult2(Z, 1, t, margin), cap_mult2(Z, 0, t, margin))
    noise_sd = 10.0 * W.std() / 1.0488
    big = D.abs() > hot_lvl * noise_sd
    p_big = 0.001 * 0.0124
    hot_c = big.sum(0, keepdim=True) >= 4 + 6 * p_big * W.shape[0]
    hot_r = big.sum(1, keepdim=True) >= 4 + 6 * p_big * W.shape[1]
    prot = (D.abs() > protect * noise_sd) & (hot_c | hot_r)
    return (Z > t) & ~prot, cm.expand_as(W), prot

def repair_all_v5(sd, t, t_wpe=6.0, verbose=False, **kw):
    out = {}; nflag = 0; ntot = 0; masks = {}
    for k, v in sd.items():
        if v.ndim != 2: out[k] = v; continue
        if k == "wpe.weight":
            R, m, m0 = repair_wpe(v, t_wpe)
        else:
            m, cm, prot = detect_v5(v, t, **kw); R = v.clone(); R[m] = cm[m]
            if verbose and prot.sum() > 0: print(f"   {k}: protected giants {int(prot.sum())}")
        if verbose: print(f"{k:28s} flagged {int(m.sum()):7d} frac {m.float().mean().item():.5f}")
        nflag += int(m.sum()); ntot += v.numel(); out[k] = R; masks[k] = m
    if verbose: print("total flagged frac", nflag / ntot)
    return out, masks
