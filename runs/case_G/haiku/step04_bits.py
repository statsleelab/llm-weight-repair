import torch, numpy as np
from safetensors import safe_open
f = safe_open("damaged.safetensors", "pt")
names = ["h.0.attn.c_proj.weight","h.7.mlp.c_fc.weight","h.8.mlp.c_fc.weight","h.1.mlp.c_proj.weight","h.5.attn.c_attn.weight","wte.weight","wpe.weight"]
for k in names:
    a = f.get_tensor(k).float().numpy().reshape(-1)
    b = a.view(np.uint32)
    low16 = (b & 0xFFFF) == 0
    low13 = (b & 0x1FFF) == 0
    # trailing-bit histogram of mantissa low byte
    lowbyte = np.bincount((b & 0xFF).astype(np.int64), minlength=256)
    exp = (b >> 23) & 0xFF
    # exponent distribution: report counts of exponents far from bulk
    eh = np.bincount(exp.astype(np.int64), minlength=256)
    nz = np.nonzero(eh)[0]
    print(k, "low16 zero frac=%.6f low13=%.6f" % (low16.mean(), low13.mean()),
          "exp range", nz.min(), nz.max(), "count exp<110 or >135:", int(eh[:110].sum()+eh[136:].sum()),
          "uniq ratio=%.4f" % (len(np.unique(b[:500000]))/500000))
    print("   lowbyte top5", np.argsort(lowbyte)[::-1][:5].tolist(), "max lowbyte count", lowbyte.max(), "mean", lowbyte.mean())
