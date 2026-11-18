import torch, sys
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_D/haiku")
from gpt2_lib import load_weights
import numpy as np
sd = load_weights("damaged.safetensors")
names = [k for k, v in sd.items() if v.dim() == 2]
print(f"{'name':32s} {'mad_sd':>7s} {'>4s':>8s} {'>6s':>8s} {'max|z|':>7s} {'kurt':>6s} {'row_max_cnt':>11s} {'col_max_cnt':>11s}")
for k in names:
    a = sd[k].numpy().astype(np.float64)
    med = np.median(a)
    mad = np.median(np.abs(a - med)) * 1.4826
    z = (a - med) / mad
    f4 = (np.abs(z) > 4).mean()
    f6 = (np.abs(z) > 6).mean()
    m = a.mean(); s = a.std()
    kurt = ((a - m) ** 4).mean() / s ** 4
    # concentration: how many rows/cols contain >=1 outlier beyond 5 sigma
    out = np.abs(z) > 5
    rows = out.sum(1); cols = out.sum(0)
    print(f"{k:32s} {mad:7.4f} {f4:8.5f} {f6:8.5f} {np.abs(z).max():7.1f} {kurt:6.2f} {rows.max():11d} {cols.max():11d}")
