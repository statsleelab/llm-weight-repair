import torch, sys
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_D/haiku")
from safetensors.torch import load_file
import numpy as np
raw = load_file("damaged.safetensors")
W = raw["wpe.weight"].numpy().astype(np.float64)
N = W.shape[0]
# residual of each row from the average of its neighbours (interior rows only)
P = np.zeros_like(W)
P[1:-1] = (W[:-2] + W[2:]) / 2
R = W - P
r = R[1:-1]
print("interior residual: std", r.std(), "MAD-sd", np.median(np.abs(r - np.median(r)))*1.4826)
qs = [0.5,0.9,0.99,0.995,0.999,0.9999]
print("|resid| quantiles", np.round(np.quantile(np.abs(r), qs),4))
print("cos(row1023,row1022)", float(W[1023]@W[1022]/np.linalg.norm(W[1023])/np.linalg.norm(W[1022])))
print("row1023 norm", np.linalg.norm(W[1023]), "row1022 norm", np.linalg.norm(W[1022]))
# per-row residual energy
e = np.sqrt((r**2).mean(1))
print("per-row residual rms quantiles 0,50,90,99,100", np.round(np.quantile(e,[0,.5,.9,.99,1]),4))
top = np.argsort(-e)[:15]
print("rows with largest residual rms:", top+1, np.round(e[top],3))
# entries with big residual
thr = 6*np.median(np.abs(r))*1.4826
bad = np.abs(R) > thr
print("threshold", round(thr,4), "n flagged", bad.sum(), "frac", bad.mean())
print("flagged per-row quantiles", np.quantile(bad.sum(1),[0,.5,.9,1]))
