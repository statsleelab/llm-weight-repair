import torch
torch.set_num_threads(1)
from safetensors.torch import load_file
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=160)
sd = load_file("damaged.safetensors")
def rstd(x):
    med = np.median(x); return np.median(np.abs(x-med))*1.4826
a = sd["wpe.weight"].numpy().astype(np.float64)
cc = np.array([rstd(c) for c in a.T])
order = np.argsort(-cc)
print("sorted col rstd (top 60):", np.round(cc[order][:60],3))
print("cols with rstd>0.05:", (cc>0.05).sum(), " >0.01:", (cc>0.01).sum(), " <0.01:", (cc<=0.01).sum())
big = cc>0.05
print("col std ratio big vs small", cc[big].mean(), cc[~big].mean())
# within a big column, what fraction of entries are 'outliers' relative to the column's core?
for j in order[:5]:
    c = a[:,j]; print("col",j, "rstd",round(cc[j],3), "quantiles", np.round(np.quantile(c,[0,.05,.25,.5,.75,.95,1]),3))
# small column structure
j = order[-1]; c=a[:,j]; print("small col", j, np.round(np.quantile(c,[0,.05,.25,.5,.75,.95,1]),4))
# per-row: fraction of big-col entries
print("first 10 positions, values of top-1 big col:", np.round(a[:10, order[0]],3))
# entries with |x|>0.1 in small columns
small = ~big
print("entries |x|>0.05 in small cols:", (np.abs(a[:,small])>0.05).sum(), "of", a[:,small].size)
print("entries |x|>0.05 in big cols:", (np.abs(a[:,big])>0.05).sum(), "of", a[:,big].size)
