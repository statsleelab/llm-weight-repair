"""Empirical-Bayes de-noising of sparsely corrupted weight matrices.
Model (established in step01-03): each 2-D matrix W = W0 + B*N, B~Bernoulli(0.01),
N ~ N(0, sig_n^2) with sig_n = 10*std(W0)  (=> std(W)^2 = 2 std(W0)^2).
Genuine entries: W0_ij = m_ij + s_ij * z,  s_ij = r_i c_j (robust rank-1 scale),
z ~ g (symmetric, non-parametric).  g is obtained by *deconvolution of counts*:
   observed |z| histogram  H = (1-pi) G + pi C,   C = known corruption histogram
Output: posterior mean E[W0 | W].   No text / loss is involved."""
import torch, numpy as np, math
PI = 0.01

def rank1_scale(R, iters=3):
    c = torch.ones(R.shape[1])
    for _ in range(iters):
        r = ((R / c).abs().median(dim=1).values * 1.4826).clamp_min(1e-8)
        c = ((R / r[:, None]).abs().median(dim=0).values * 1.4826).clamp_min(1e-8)
    return r, c

def local_scale(R, key="", k=8, seed=0):
    """Scale model for genuine entries.  rank-1 robust scale, times explicit attention-head block
    factors (64-wide blocks of c_attn outputs / attn.c_proj inputs), times a rank-k model of the
    residual log-magnitude log|z| (captures row-group x column-group scale structure).
    All pieces are medians / low-rank fits over >= 64 entries so a single corrupted entry cannot
    raise its own scale."""
    r, c = rank1_scale(R)
    S = r[:, None] * c[None, :]
    if key.endswith("attn.c_attn.weight"):
        Z = (R / S).abs().view(R.shape[0], -1, 64)
        b = (Z.median(dim=2).values * 1.4826).clamp_min(1e-3)
        S = S * b.repeat_interleave(64, dim=1)
    elif key.endswith("attn.c_proj.weight"):
        Z = (R / S).abs().view(-1, 64, R.shape[1])
        b = (Z.median(dim=1).values * 1.4826).clamp_min(1e-3)
        S = S * b.repeat_interleave(64, dim=0)
    if k > 0:
        E = torch.log((R / S).abs().clamp(1e-3, 6.0))       # winsorised log-magnitude
        E = E - E.mean()
        E = E - E.mean(1, keepdim=True); E = E - E.mean(0, keepdim=True)
        torch.manual_seed(seed)
        U, sv, V = torch.svd_lowrank(E, q=k + 8, niter=3)
        F = (U[:, :k] * sv[:k]) @ V[:, :k].T
        S = S * torch.exp(F)
        del E, F
    S = S * ((R / S).abs().median() * 1.4826)
    return S

def make_edges():
    fine = torch.sinh(torch.linspace(0, math.asinh(4.0), 120))
    coarse = torch.tensor([4.5, 5, 5.5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 1e9])
    return torch.cat([fine, coarse])

def Phi(x): return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

CEDGES = torch.tensor([0, 2, 3, 4, 5, 6, 8, 11, 15, 20, 30, 50.0])

def line_density(R, S, sig_n, G, edges, dim, ksig=3.0, add=2.0):
    """Line-specific (row: dim=1, column: dim=0) heavy-tail correction in *local z units*.
    For each line and coarse |z| bin: H = observed count, C = expected # corrupted entries there
    (exact, from each entry's own scale), Gx = expected # genuine entries under the global shape g.
    Where H significantly exceeds C+Gx the line genuinely has a heavier tail: its genuine density
    in that bin is (H-C)/(n * 2 * width).  Returns per-entry density (0 where no correction)."""
    nb = len(CEDGES) - 1
    Z = (R / S).abs()
    cb = torch.bucketize(Z, CEDGES[1:])              # 0..nb  (nb = beyond 50: never corrected)
    del Z
    nl = R.shape[0] if dim == 1 else R.shape[1]
    n = R.shape[1] if dim == 1 else R.shape[0]
    line = (torch.arange(R.shape[0])[:, None] if dim == 1 else torch.arange(R.shape[1])[None, :])
    idx = line * (nb + 1) + cb
    H = torch.bincount(idx.flatten(), minlength=nl * (nb + 1)).float().view(nl, nb + 1)[:, :nb]
    ratio = S / torch.sqrt(sig_n ** 2 + S ** 2)
    C = torch.zeros(nl, nb)
    prev = torch.zeros_like(ratio)
    for b in range(nb):
        cur = 2 * Phi(CEDGES[b + 1] * ratio) - 1
        C[:, b] = PI * (cur - prev).sum(dim)
        prev = cur
    # global genuine fraction per coarse bin
    cum = torch.cat([torch.zeros(1, dtype=G.dtype), torch.cumsum(G, 0)]) / G.sum()
    pos = torch.bucketize(CEDGES, edges)             # CEDGES are a subset of fine edges for >=4; approx below
    Gfrac = torch.stack([cum[min(pos[b + 1], len(cum) - 1)] - cum[min(pos[b], len(cum) - 1)] for b in range(nb)]).float()
    Gx = (1 - PI) * n * Gfrac[None, :]
    E = C + Gx
    sig = H > E + ksig * torch.sqrt(E) + add
    width = (CEDGES[1:] - CEDGES[:-1])[None, :]
    dens = torch.where(sig, (H - C) / ((1 - PI) * n * 2 * width), torch.zeros_like(H))
    dens = torch.cat([dens, torch.zeros(nl, 1)], 1)
    return dens.flatten()[idx]

AEDGES = torch.tensor([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.5])

def line_cap(A, dim, k=4.0, add=3.0):
    """Per-line (row if dim=1, column if dim=0) count deconvolution in absolute units A=|R|/sig_n.
    In every (line, |value|-bin) cell the expected number of corrupted entries is C = pi*n*P(bin).
    Where the observed count H is significantly larger (H > C + k sqrt(C) + add) the excess must be
    genuine structure of that line, so P(corrupt) <= C/H there.  Returns the cap (1 = no cap)."""
    nb = len(AEDGES) + 1
    b = torch.bucketize(A, AEDGES)
    nl = A.shape[1 - dim] if dim == 0 else A.shape[0]
    n = A.shape[dim] if dim == 0 else A.shape[1]
    line = (torch.arange(A.shape[0])[:, None] if dim == 1 else torch.arange(A.shape[1])[None, :])
    idx = (line * nb + b)
    H = torch.bincount(idx.flatten(), minlength=nl * nb).float().view(nl, nb)
    e = torch.cat([torch.zeros(1), AEDGES, torch.tensor([1e6])])
    P = 2 * (Phi(e[1:]) - Phi(e[:-1]))
    C = (PI * n * P)[None, :]
    cap = torch.where(H > C + k * torch.sqrt(C) + add, C / H.clamp_min(1), torch.ones_like(H))
    return cap.flatten()[idx]

def posterior(W, m=None, sig_n=None, verbose=False, S=None, seed=0, line_guard=False, key='', k=8, line_z=True, line_dims=(0, 1), ksig=5.0, add=4.0):
    """returns (W_hat, p_corrupt)."""
    if m is None: m = torch.zeros(1)
    if sig_n is None: sig_n = 10 * W.std().item() / math.sqrt(2)
    R = W - m
    N = R.numel()
    if S is None:
        S = local_scale(R, key, k)
    edges = make_edges(); nb = len(edges) - 1
    widths = edges[1:] - edges[:-1]
    bi = torch.bucketize((R / S).abs(), edges[1:-1])
    H = torch.bincount(bi.flatten(), minlength=nb).double()
    # expected corrupted counts per z-bin: |v-m| ~ |N(0, sig_n^2+S^2)|, averaged over a sample of S
    gen = torch.Generator().manual_seed(seed)
    samp = S.flatten()[torch.randint(0, N, (100000,), generator=gen)].double()
    ratio = samp / torch.sqrt(sig_n ** 2 + samp ** 2)                    # z-edge -> std-normal units
    cdf = 2 * Phi(edges[None, :].double().clamp_max(1e6) * ratio[:, None]) - 1
    cdf[:, -1] = 1.0
    C = PI * N * (cdf[:, 1:] - cdf[:, :-1]).mean(0)
    G = (H - C).clamp_min(0)
    g = (G / G.sum() / widths.double() / 2).float()
    g = torch.cummin(g, 0).values
    var_c = sig_n ** 2 + S ** 2
    cd = PI * torch.exp(-0.5 * R ** 2 / var_c) / torch.sqrt(2 * math.pi * var_c)
    gz = g[bi]
    if line_z:
        for d in line_dims:
            gz = torch.maximum(gz, line_density(R, S, sig_n, G, edges, d, ksig, add))
    gd = (1 - PI) * gz / S
    p = cd / (cd + gd + 1e-30)
    # guard for genuine giant weights (absolute units): impossible / over-represented as Gaussian noise
    A = R.abs() / sig_n
    aedges = [3.0, 3.5, 4.0, 4.5, 5.5]
    for lo, hi in zip(aedges[:-1], aedges[1:]):
        M = (A >= lo) & (A < hi)
        obs = M.sum().item()
        exp = PI * N * 2 * (Phi(torch.tensor(hi)) - Phi(torch.tensor(lo))).item()
        if obs > 3 * exp + 3:
            p[M] = torch.minimum(p[M], torch.tensor(exp / obs))
            if verbose: print(f"   abs guard [{lo},{hi}) sig_n: obs {obs} exp {exp:.1f}")
    p[A >= 5.5] = 0.0
    if line_guard:
        p = torch.minimum(p, line_cap(A, 1)); p = torch.minimum(p, line_cap(A, 0))
    shrink = S ** 2 / var_c
    What = m + (1 - p) * R + p * shrink * R
    if verbose: print("   E[#corrupt]/N = %.5f  frac p>.5 = %.5f" % (p.mean().item(), (p > .5).float().mean().item()))
    return What, p
