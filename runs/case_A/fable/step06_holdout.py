import torch, sys, time
torch.set_num_threads(1)
from gpt2_lib import load_weights
from impute_lib import *
sd = load_weights("damaged.safetensors")
keys = sys.argv[1:] 
g = torch.Generator().manual_seed(0)
for k in keys:
    X = sd[k]; t0=time.time()
    miss0 = X == 0
    H = (torch.rand(X.shape, generator=g) < 0.01) & ~miss0
    miss = miss0 | H
    Xc, mu = center(X, miss)
    true = (X - mu)[H]
    base = (true**2).sum()
    U,S,Vh = torch.linalg.svd(Xc, full_matrices=False)
    res = {}
    preds = {}
    for lam in [0.01, 0.03, 0.1, 0.3, 1.0]:
        pa = cond_impute(Xc, miss, lam, S, Vh.T)[H]
        res[("A",lam)] = 1 - ((pa-true)**2).sum()/base; preds[("A",lam)] = pa
        if X.shape[1] <= 4000 and X.shape[0] <= 4000:
            pb = cond_impute(Xc.T.contiguous(), miss.T.contiguous(), lam, S, U).T[H]
            res[("B",lam)] = 1 - ((pb-true)**2).sum()/base; preds[("B",lam)] = pb
    ba = max((kk for kk in res if kk[0]=="A"), key=lambda kk: res[kk])
    line = f"{k:26s} raw-var-share-of-mean-model={1-(base/(X[H]**2).sum()).item():.3f} A:" + " ".join(f"{res[('A',l)].item():.3f}" for l in [0.01,0.03,0.1,0.3,1.0])
    if ("B",0.1) in res:
        bb = max((kk for kk in res if kk[0]=="B"), key=lambda kk: res[kk])
        avg = (preds[ba]+preds[bb])/2
        line += " B:" + " ".join(f"{res[('B',l)].item():.3f}" for l in [0.01,0.03,0.1,0.3,1.0]) + f" avg(bestA,bestB)={(1-((avg-true)**2).sum()/base).item():.3f}"
    print(line, f"[{time.time()-t0:.0f}s]", flush=True)
