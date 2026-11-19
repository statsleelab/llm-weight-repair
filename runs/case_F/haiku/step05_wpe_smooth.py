import torch
torch.set_num_threads(1)
from safetensors.torch import load_file
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=160)
sd = load_file("damaged.safetensors")
a = sd["wpe.weight"].numpy().astype(np.float64)
def rstd(x):
    med = np.median(x); return np.median(np.abs(x-med))*1.4826
cc = np.array([rstd(c) for c in a.T])
big = cc>0.05
# lag-1 smoothness: residual from average of neighbours
nb = np.full_like(a, np.nan)
nb[1:-1] = 0.5*(a[:-2]+a[2:])
res = a - nb
res_big = res[1:-1][:, big]; res_small = res[1:-1][:, ~big]
print("median |res| big cols", np.median(np.abs(res_big)), "small cols", np.median(np.abs(res_small)))
print("q99 |res| big", np.quantile(np.abs(res_big),[.5,.9,.99]), "small", np.quantile(np.abs(res_small),[.5,.9,.99,.999]))
# look at a small column's largest entries and their neighbours
j = 10
col = a[:,j]
idx = np.argsort(-np.abs(col))[:8]
print("small col example", j, "top idx", idx, "vals", np.round(col[idx],3))
for i in idx[:3]:
    print("  around", i, np.round(col[max(0,i-3):i+4],3))
# distribution of values in small columns core: is there a sparse mixture?
s = a[:, ~big].ravel()
print("small cols quantiles", np.round(np.quantile(s,[.001,.01,.03,.05,.5,.95,.97,.99,.999]),4))
