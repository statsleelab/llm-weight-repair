import torch, time
torch.set_num_threads(1)
from gpt2_lib import load_weights
from impute_lib import *
sd = load_weights("damaged.safetensors")
X = sd["wte.weight"]; g = torch.Generator().manual_seed(1)
miss0 = X == 0
H = (torch.rand(X.shape, generator=g) < 0.01) & ~miss0
miss = miss0 | H
Xc, mu = center(X, miss)
true = (X-mu)[H]; base=(true**2).sum()
r2 = lambda p: (1 - ((p-true)**2).sum()/base).item()
U,S,Vh = torch.linalg.svd(Xc, full_matrices=False)
rd = cond_impute(Xc, miss, 0.03, S, Vh.T)
print("ridge", r2(rd[H]), flush=True)
Xf = Xc + rd*miss
Xn = Xf / Xf.norm(dim=1, keepdim=True)
obs = (~miss).float()
KM=32
V = torch.zeros(X.shape[0], KM); IX = torch.zeros(X.shape[0], KM, dtype=torch.long)
for s in range(0, X.shape[0], 1024):
    sim = Xn[s:s+1024] @ Xn.T
    sim[torch.arange(sim.shape[0]), torch.arange(s, s+sim.shape[0])] = -1
    V[s:s+1024], IX[s:s+1024] = sim.topk(KM, dim=1)
print("sim quantiles of nn1:", [round(float(V[:,0].quantile(q)),3) for q in (.1,.5,.9)], "nn8:", [round(float(V[:,7].quantile(q)),3) for q in (.1,.5,.9)], flush=True)
def knn(K, p):
    w = V[:, :K].clamp(min=1e-6).pow(p)[:, :, None]
    M = torch.zeros_like(Xc)
    for s in range(0, X.shape[0], 4096):
        nb = Xc[IX[s:s+4096, :K]]; ob = obs[IX[s:s+4096, :K]]
        M[s:s+4096] = (w[s:s+4096]*nb*ob).sum(1) / (w[s:s+4096]*ob).sum(1).clamp(min=1e-6)
    return M
best=None
for K in [4,8,16]:
    for p in [1,4,12]:
        M = knn(K,p); r=r2(M[H])
        print(f"kNN K={K} p={p} R2={r:.4f}", flush=True)
        if best is None or r>best[0]: best=(r,K,p,M)
_,K,p,M = best
print("best",K,p)
for a in [0.6,0.8,1.0]:
    # local model: residual wrt shrunk neighbour mean, then global ridge on the residual
    R = (Xc - a*M) * obs
    for lam in [0.03, 0.3]:
        U2,S2,Vh2 = torch.linalg.svd(R, full_matrices=False)
        pr = a*M + cond_impute(R, miss, lam, S2, Vh2.T)
        print(f"kNN-mean*{a} + ridge-on-residual lam={lam}: R2={r2(pr[H]):.4f}", flush=True)
# column-wise view: where is the remaining error, for the big-mean columns
cm = mu.mean(0); top = cm.abs().topk(8).indices
Hc = H.float()
for j in top.tolist():
    h = H[:, j]; t = (X-mu)[h, j]
    print(f"col {j}: colmean {cm[j]:.3f} resid-std {t.std():.3f} R2 ridge {1-((rd[h,j]-t)**2).sum()/(t**2).sum():.3f} knn {1-((M[h,j]-t)**2).sum()/(t**2).sum():.3f}")
