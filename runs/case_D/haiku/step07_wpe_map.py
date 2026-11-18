import torch, sys
torch.set_num_threads(1)
from safetensors.torch import load_file
import numpy as np
raw = load_file("damaged.safetensors")
W = raw["wpe.weight"].numpy().astype(np.float64)
P = np.zeros_like(W); P[1:-1] = (W[:-2] + W[2:]) / 2
R = W - P
r = R[1:-1]
thr = 0.0237
bad = np.abs(R) > thr
rowc = bad.sum(1)
print("flagged per 32-row block:")
for b in range(0, 1024, 32):
    print(f"{b:5d}: {rowc[b:b+32].sum():5d}  " + " ".join(str(x) for x in rowc[b:b+32]))
colc = bad.sum(0)
print("col counts quantiles", np.quantile(colc,[0,.5,.9,1]))
