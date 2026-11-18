import torch, sys, time, json
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights
from impute_lib import *
sd = load_weights("damaged.safetensors")
g = torch.Generator().manual_seed(0)
LAMS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
out = dict(sd); log = {}
for k in sorted(sd):
    X = sd[k]
    if X.ndim != 2: continue
    t0 = time.time()
    miss0 = X == 0
    H = (torch.rand(X.shape, generator=g) < 0.01) & ~miss0
    miss = miss0 | H
    Xc, mu = center(X, miss)
    true = (X - mu)[H]; base = (true**2).sum()
    U,S,Vh = torch.linalg.svd(Xc, full_matrices=False)
    res = {}; preds = {}
    small = X.shape[0] <= 4000
    for lam in LAMS:
        preds[("A",lam)] = cond_impute(Xc, miss, lam, S, Vh.T)[H]
        if small: preds[("B",lam)] = cond_impute(Xc.T.contiguous(), miss.T.contiguous(), lam, S, U).T[H]
    r2 = lambda p: (1 - ((p-true)**2).sum()/base).item()
    for kk in preds: res[kk] = r2(preds[kk])
    ba = max((kk for kk in res if kk[0]=="A"), key=res.get)
    cands = {"A": res[ba]}
    if small:
        bb = max((kk for kk in res if kk[0]=="B"), key=res.get)
        cands["B"] = res[bb]; cands["AB"] = r2((preds[ba]+preds[bb])/2)
    mode = max(cands, key=cands.get)
    # R2 of the mean model alone vs raw zero baseline
    r2mean = 1 - (base/(X[H]**2).sum()).item()
    # final imputation with the true mask only
    Xc, mu = center(X, miss0)
    U,S,Vh = torch.linalg.svd(Xc, full_matrices=False)
    P = torch.zeros_like(X); n=0
    if cands[mode] > 0:
        if "A" in mode: P += cond_impute(Xc, miss0, ba[1], S, Vh.T); n+=1
        if "B" in mode: P += cond_impute(Xc.T.contiguous(), miss0.T.contiguous(), bb[1], S, U).T; n+=1
        P /= n
    fill = (mu + P) if (r2mean > 0 or cands[mode] > 0) else torch.zeros_like(X)
    Y = X.clone(); Y[miss0] = fill[miss0]
    out[k] = Y
    log[k] = dict(mode=mode, r2=cands[mode], lamA=ba[1], lamB=(bb[1] if small else None), r2mean=r2mean, n_fixed=int(miss0.sum()))
    print(f"{k:26s} mode={mode:2s} holdoutR2={cands[mode]:.3f} lamA={ba[1]} lamB={bb[1] if small else None} r2mean={r2mean:.3f} [{time.time()-t0:.0f}s]", flush=True)
save_weights(out, "cand_v1.safetensors")
json.dump(log, open("step07_log.json","w"), indent=1)
