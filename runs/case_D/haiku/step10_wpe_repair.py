"""Repair wpe.weight using its strong smoothness across positions.

For each dimension j, the position sequence x_i = wpe[i, j] is smooth.  A
corrupted entry shows up as a large second difference d_i = x_i - (x_{i-1}+x_{i+1})/2.
We estimate the local (per-block, per-dim) noise scale robustly from the median
absolute second difference, flag |d_i| > K * scale, and replace flagged entries
with the interpolation from their (unflagged) neighbours.  Two passes.
"""
import torch, sys
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_D/haiku")
from safetensors.torch import load_file
import numpy as np

K = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
BLOCK = 64

def repair_wpe(W, K=K, passes=2):
    W = W.copy()
    n = W.shape[0]
    flagged_total = np.zeros_like(W, dtype=bool)
    for p in range(passes):
        d = np.zeros_like(W)
        d[1:-1] = W[1:-1] - 0.5 * (W[:-2] + W[2:])
        flag = np.zeros_like(W, dtype=bool)
        for b in range(0, n, BLOCK):
            sl = slice(max(b,1), min(b+BLOCK, n-1))
            # robust scale per dim over block (from second differences)
            s = 1.4826 * np.median(np.abs(d[sl] - np.median(d[sl], axis=0)), axis=0) + 1e-6
            flag[sl] = np.abs(d[sl]) > K * s
        # also flag rows 0 and n-1 never (edges left alone)
        flag[0] = False; flag[-1] = False
        flagged_total |= flag
        # impute flagged entries from unflagged neighbours
        new = W.copy()
        idx = np.argwhere(flag)
        for i, j in idx:
            l = i - 1
            r = i + 1
            while l >= 0 and flag[l, j]: l -= 1
            while r < n and flag[r, j]: r += 1
            if l < 0 and r >= n: continue
            if l < 0: new[i, j] = W[r, j]
            elif r >= n: new[i, j] = W[l, j]
            else:
                w_l = (r - i) / (r - l); w_r = (i - l) / (r - l)
                new[i, j] = w_l * W[l, j] + w_r * W[r, j]
        W = new
    return W, flagged_total

if __name__ == "__main__":
    raw = load_file("damaged.safetensors")
    W0 = raw["wpe.weight"].numpy().astype(np.float64)
    W1, flag = repair_wpe(W0)
    print("K", K, "flagged entries", int(flag.sum()), "frac", round(float(flag.mean()),5))
    print("flagged per row-block of 128:", [int(flag[b:b+128].sum()) for b in range(0,1024,128)])
    print("row-1023 norm before/after", np.linalg.norm(W0[1023]), np.linalg.norm(W1[1023]))
    np.save("wpe_repaired_K%s.npy" % K, W1.astype(np.float32))
