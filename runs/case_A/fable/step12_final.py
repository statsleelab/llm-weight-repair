"""Final repair: h.* and wpe from cand_v1 (ridge/Gaussian-conditional imputation),
wte re-imputed with kNN-token-mean + ridge on the residual. Holdout check first."""
import torch, time
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights
from impute_lib import *
sd = load_weights("damaged.safetensors")
X = sd["wte.weight"]; miss0 = X == 0
LAM = 0.03; KM = 32

def wte_impute(miss, configs):
    """returns {cfg: full prediction matrix (in original units)}, plus mu"""
    Xc, mu = center(X, miss); obs = (~miss).float()
    U,S,Vh = torch.linalg.svd(Xc, full_matrices=False)
    rd = cond_impute(Xc, miss, LAM, S, Vh.T)
    Xf = Xc + rd*miss
    Xn = Xf / Xf.norm(dim=1, keepdim=True)
    V = torch.zeros(X.shape[0], KM); IX = torch.zeros(X.shape[0], KM, dtype=torch.long)
    for s in range(0, X.shape[0], 1024):
        sim = Xn[s:s+1024] @ Xn.T
        sim[torch.arange(sim.shape[0]), torch.arange(s, s+sim.shape[0])] = -1
        V[s:s+1024], IX[s:s+1024] = sim.topk(KM, dim=1)
    out = {"ridge": mu + rd}
    for (K,p) in configs:
        w = V[:, :K].clamp(min=1e-6).pow(p)[:, :, None]
        M = torch.zeros_like(Xc)
        for s in range(0, X.shape[0], 4096):
            nb = Xc[IX[s:s+4096, :K]]; ob = obs[IX[s:s+4096, :K]]
            M[s:s+4096] = (w[s:s+4096]*nb*ob).sum(1) / (w[s:s+4096]*ob).sum(1).clamp(min=1e-6)
        R = (Xc - M) * obs
        U2,S2,Vh2 = torch.linalg.svd(R, full_matrices=False)
        out[(K,p)] = mu + M + cond_impute(R, miss, LAM, S2, Vh2.T)
    return out, mu

g = torch.Generator().manual_seed(2)
H = (torch.rand(X.shape, generator=g) < 0.01) & ~miss0
cfgs = [(16,4),(32,4),(32,6)]
preds, mu = wte_impute(miss0 | H, cfgs)
true = X[H]; base = ((X-mu)[H]**2).sum()
score = {}
for c,P in preds.items():
    score[c] = (1 - ((P[H]-true)**2).sum()/base).item()
    line = f"holdout {c}: R2(centred)={score[c]:.4f}  R2(vs zero)={(1-((P[H]-true)**2).sum()/(true**2).sum()).item():.4f} | per-col R2:"
    for j in [36,64,87,138,266,326,362,6,496,430,314]:
        h = H[:, j]; t = X[h, j]-mu[h, j]
        line += f" {j}:{(1-((P[h,j]-X[h,j])**2).sum()/(t**2).sum()).item():.2f}"
    print(line, flush=True)
best = max(cfgs, key=lambda c: score[c]); print("chosen", best, flush=True)
del preds
preds, mu = wte_impute(miss0, [best])
out = load_weights("cand_v1.safetensors")
W = X.clone(); W[miss0] = preds[best][miss0]
out["wte.weight"] = W
# sanity: identical to damaged everywhere except at zeroed entries of 2-D weights
for k,v in sd.items():
    assert out[k].shape == v.shape and torch.isfinite(out[k]).all()
    if v.ndim == 2: assert torch.equal(out[k][v != 0], v[v != 0])
    else: assert torch.equal(out[k], v)
save_weights(out, "repaired.safetensors"); print("saved", flush=True)
