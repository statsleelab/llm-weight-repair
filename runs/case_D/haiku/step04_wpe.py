import torch, sys
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_D/haiku")
from safetensors.torch import load_file
import numpy as np
raw = load_file("damaged.safetensors")
W = raw["wpe.weight"].numpy().astype(np.float64)
rn = np.linalg.norm(W, axis=1)
print("row L2 norm, positions 0..15:", np.round(rn[:16],2))
print("row L2 norm quantiles 0,10,50,90,99,100:", np.round(np.quantile(rn,[0,.1,.5,.9,.99,1]),3))
print("row L2 norm at positions 100,200,...,1000:", np.round(rn[100::100],2))
big = np.abs(W) > 0.3
print("entries |w|>0.3 total:", big.sum(), "frac", big.mean())
cnt_row = big.sum(1)
print("per-row count of |w|>0.3, quantiles 0,50,90,99,100:", np.quantile(cnt_row,[0,.5,.9,.99,1]))
print("first 20 rows counts:", cnt_row[:20])
print("per-col count of |w|>0.3 top 10 cols:", np.argsort(-big.sum(0))[:10], np.sort(-big.sum(0))[:10]*-1)
# histogram of row norms
h, e = np.histogram(rn, bins=20)
print("row-norm hist:", h.tolist()); print("edges", np.round(e,2).tolist())
