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
ridge = {}
for lam in [0.001,0.003,0.01,0.03,0.1]:
    ridge[lam] = cond_impute(Xc, miss, lam, S, Vh.T)
    print("ridge lam",lam,"R2",r2(ridge[lam][H]),flush=True)
bl = min(ridge, key=lambda l: -r2(ridge[l][H]))
# EM second pass
X1 = Xc + ridge[bl]*miss
U1,S1,Vh1 = torch.linalg.svd(X1, full_matrices=False)
em = cond_impute(Xc, miss, bl, S1, Vh1.T); print("EM pass2 R2", r2(em[H]), flush=True)
# kNN on ridge-filled centred matrix (cosine)
Xf = Xc + em*miss
Xn = Xf / Xf.norm(dim=1, keepdim=True)
obs = (~miss).float()
t0=time.time()
for K in [8, 32]:
    pred = torch.zeros_like(Xc)
    rows = miss.any(1).nonzero().squeeze(1)
    for s in range(0, len(rows), 1024):
        r = rows[s:s+1024]
        sim = Xn[r] @ Xn.T
        sim[torch.arange(len(r)), r] = -1
        v, ix = sim.topk(K, dim=1)
        w = v.clamp(min=0)[:, :, None]                       # (b,K,1)
        nb = Xc[ix]; ob = obs[ix]                             # neighbours' observed values only
        pred[r] = (w*nb*ob).sum(1) / (w*ob).sum(1).clamp(min=1e-6)
    print("kNN K",K,"R2",r2(pred[H]), "[%.0fs]"%(time.time()-t0), flush=True)
    for a in [0.25,0.5,0.75]:
        print("   blend knn*%.2f + ridge*(1-a): R2 %.4f"%(a, r2(a*pred[H]+(1-a)*em[H])))
    # residual kNN: neighbours' ridge residuals
