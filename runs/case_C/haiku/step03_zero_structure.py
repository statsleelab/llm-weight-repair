# Inspect the structure of the exact-zero mask: histogram of per-row counts, positions in rows with many zeros.
import sys
sys.path.insert(0, "/home/user/runs/case_C/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights
import collections
sd = load_weights("/home/user/runs/case_C/haiku/damaged.safetensors")
for k in ["wte.weight", "h.0.attn.c_attn.weight", "h.0.mlp.c_fc.weight", "h.0.attn.c_proj.weight"]:
    Z = (sd[k] == 0)
    rc = Z.sum(1)
    hist = collections.Counter(rc.tolist())
    print(k, "rows:", Z.shape[0], "row-zero-count histogram (count:numRows), top:",
          sorted(hist.items(), key=lambda kv: -kv[1])[:12])
    # which rows have many zeros, and are their zeros contiguous / spread?
    idx = torch.argsort(rc, descending=True)[:3].tolist()
    for i in idx:
        pos = torch.nonzero(Z[i]).flatten()
        print("   row", i, "n_zero", len(pos), "first positions", pos[:20].tolist(), "last", pos[-5:].tolist())
    # column view
    cc = Z.sum(0)
    histc = collections.Counter(cc.tolist())
    print("   col-zero-count histogram top:", sorted(histc.items(), key=lambda kv: -kv[1])[:8])
