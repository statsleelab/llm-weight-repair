import torch
def gauss_impute(W, miss, lam=0.1, rounds=3, orient=None):
    """Conditional-Gaussian (EM-style) imputation. Samples = rows of A, features = cols.
    orient: 'rows' -> samples are rows of W; 'cols' -> samples are columns of W."""
    if orient is None: orient = 'rows' if W.shape[0] >= W.shape[1] else 'cols'
    A = W.clone().double(); M = miss.clone()
    if orient == 'cols': A = A.T.contiguous(); M = M.T.contiguous()
    n, f = A.shape
    obs = (~M).double()
    mu = (A * obs).sum(0) / obs.sum(0)
    A[M] = mu.expand_as(A)[M]
    rows = M.any(1).nonzero().flatten().tolist()
    for _ in range(rounds):
        mu = A.mean(0)
        C = A - mu
        S = C.T @ C / n
        S = S + lam * S.diagonal().mean() * torch.eye(f, dtype=S.dtype)
        P = torch.linalg.inv(S)
        for r in rows:
            m = M[r]; mi = m.nonzero().flatten(); oi = (~m).nonzero().flatten()
            Pmm = P[mi][:, mi]; Pmo = P[mi][:, oi]
            A[r, mi] = mu[mi] - torch.linalg.solve(Pmm, Pmo @ (A[r, oi] - mu[oi]))
    out = A.float()
    if orient == 'cols': out = out.T.contiguous()
    R = W.clone(); R[miss] = out[miss]; return R
def lowrank_impute(W, miss, k, iters=15):
    X = W.clone(); X[miss] = 0
    for _ in range(iters):
        U, S, V = torch.svd_lowrank(X, q=k + 10, niter=2)
        X[miss] = ((U[:, :k] * S[:k]) @ V[:, :k].T)[miss]
    return X
