import torch
from repair_lib import wpe_resid
def robust_zc(v):
    cm = v.median(0, keepdim=True).values
    u = v - cm
    c = u.abs().median(0, keepdim=True).values/0.6745 + 1e-8
    u = u / c
    rm = u.median(1, keepdim=True).values
    u = u - rm
    r = u.abs().median(1, keepdim=True).values/0.6745 + 1e-8
    return u / r, cm
def detect(v, T, heavy_k=4.0):
    z, cm = robust_zc(v)
    big = z.abs() > 6
    n_r, n_c = v.shape
    hc = big.sum(0) > (4 + heavy_k*0.0006*n_r)
    hr = big.sum(1) > (4 + heavy_k*0.0006*n_c)
    m = (z.abs() > T) & ~(hr[:, None] | hc[None, :])
    s = v.abs().median()/0.6745
    m &= v.abs() < 50*s
    return m, hr, hc, cm
def repair(sd, T=6.0, Twpe=12.0, verbose=False):
    out = dict(sd); masks = {}
    for k, v in sd.items():
        if v.dim() != 2: continue
        m, hr, hc, cm = detect(v, T)
        fill = cm.expand_as(v)
        if k == "wpe.weight":
            zr, med = wpe_resid(v)
            ms = zr.abs() > Twpe; ms[:3] = False
            m = m.clone(); m[3:] = False
            nv = torch.where(ms, med, torch.where(m, fill, v)); m = m | ms
        else:
            nv = torch.where(m, fill, v)
        out[k] = nv; masks[k] = m
        if verbose: print(f"{k:28s} flagged {int(m.sum()):6d} (0.1%={v.numel()//1000}) heavy rows {int(hr.sum())} cols {int(hc.sum())}")
    return out, masks
