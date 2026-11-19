import torch
from repair_lib import robust_z, wpe_resid
def detect(v, T, heavy_k=4.0):
    z = robust_z(v)
    big = z.abs() > 6
    n_r, n_c = v.shape
    hc = big.sum(0) > (4 + heavy_k*0.0006*n_r)   # heavy columns (structured large weights)
    hr = big.sum(1) > (4 + heavy_k*0.0006*n_c)   # heavy rows
    m = z.abs() > T
    m &= ~(hr[:, None] | hc[None, :])
    # corrupted values never exceed ~45 tensor-sigmas: anything larger is genuine
    s = v.abs().median()/0.6745
    m &= v.abs() < 50*s
    return m, hr, hc
def repair(sd, T=6.0, Twpe=12.0, verbose=False):
    out = dict(sd); masks = {}
    for k, v in sd.items():
        if v.dim() != 2: continue
        m, hr, hc = detect(v, T)
        fill = v.median(0, keepdim=True).values.expand_as(v)
        if k == "wpe.weight":
            zr, med = wpe_resid(v)
            ms = zr.abs() > Twpe
            ms[:3] = False
            m = m.clone(); m[3:] = False
            nv = torch.where(m, fill, v)
            nv = torch.where(ms, med, nv)
            m = m | ms
        else:
            nv = torch.where(m, fill, v)
        out[k] = nv; masks[k] = m
        if verbose: print(f"{k:28s} flagged {int(m.sum()):6d} (0.1%={v.numel()//1000}) heavy rows {int(hr.sum())} cols {int(hc.sum())}")
    return out, masks
