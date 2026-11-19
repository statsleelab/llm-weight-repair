import torch
from repair_lib import robust_z, wpe_resid
def heavy(v, thr=6.0, mult=8.0):
    s = v.abs().median()/0.6745
    big = (v.abs() > thr*s)
    exp_c = 0.0006*v.shape[0]; exp_r = 0.0006*v.shape[1]
    hc = big.sum(0) > (4 + mult*exp_c)
    hr = big.sum(1) > (4 + mult*exp_r)
    return hr, hc
def repair(sd, T=8.0, Twpe=8.0, protect=True, verbose=False):
    out = dict(sd); masks = {}
    for k, v in sd.items():
        if v.dim() != 2: continue
        if k == "wpe.weight":
            z, med = wpe_resid(v)
            m = z.abs() > Twpe
            m[0] = False
            nv = torch.where(m, med, v)
        else:
            # global robust scale (not column) so genuine heavy columns aren't normalised away,
            s = v.abs().median()/0.6745
            z = v / s
            m = z.abs() > T
            if protect:
                hr, hc = heavy(v)
                m &= ~(hr[:, None] | hc[None, :])
            fill = v.median(0, keepdim=True).values.expand_as(v) if k == "wte.weight" else torch.zeros_like(v)
            nv = torch.where(m, fill, v)
            if verbose: print(k, "heavy rows", hr.nonzero().flatten().tolist()[:20], "heavy cols", hc.nonzero().flatten().tolist()[:20])
        out[k] = nv; masks[k] = m
        if verbose: print("  ", k, int(m.sum()), "flagged; 0.1% =", v.numel()//1000)
    return out, masks
