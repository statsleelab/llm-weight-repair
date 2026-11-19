import torch
def robust_z(v):
    c = v.abs().median(0, keepdim=True).values/0.6745 + 1e-8   # column scale
    u = v / c
    r = u.abs().median(1, keepdim=True).values/0.6745 + 1e-8   # row scale
    return u / r
def wpe_resid(v):
    # residual from median of 4 neighbouring positions
    P = v.shape[0]
    pad = torch.cat([v[:1].repeat(2,1)*float('nan'), v, v[-1:].repeat(2,1)*float('nan')])
    nb = torch.stack([pad[0:P], pad[1:P+1], pad[3:P+3], pad[4:P+4]])
    med = nb.nanmedian(0).values
    res = v - med
    s = res.abs().median(0, keepdim=True).values/0.6745 + 1e-8
    return res/s, med
def repair(sd, T=8.0, Twpe=8.0, verbose=False):
    out = dict(sd); masks = {}
    for k, v in sd.items():
        if v.dim() != 2: continue
        if k == "wpe.weight":
            z, med = wpe_resid(v)
            m = z.abs() > Twpe
            m[0] = robust_z(v)[0].abs() > 1e9  # don't touch row 0 via smoothness
            nv = torch.where(m, med, v)
        else:
            z = robust_z(v)
            m = z.abs() > T
            fill = v.median(0, keepdim=True).values.expand_as(v) if k == "wte.weight" else torch.zeros_like(v)
            nv = torch.where(m, fill, v)
        out[k] = nv; masks[k] = m
        if verbose: print(k, int(m.sum()), "of", v.numel()//1000, "expected")
    return out, masks
