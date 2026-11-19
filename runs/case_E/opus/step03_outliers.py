import torch; torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
def lz(v):
    r = v.abs().median(1,keepdim=True).values
    c = v.abs().median(0,keepdim=True).values
    return v / (r*c/ v.abs().median()) * 0.6745
for k in ["wpe.weight","h.5.attn.c_proj.weight","h.7.mlp.c_fc.weight","wte.weight"]:
    v=sd[k]; z=lz(v); s=v.std()
    m = z.abs()>8
    vals=(v[m]/s)
    print(k, m.sum().item(), "outlier |v|/std quantiles:", [round(x,2) for x in vals.abs().quantile(torch.tensor([0,.1,.25,.5,.75,.9,1.])).tolist()], "pos frac",(vals>0).float().mean().item())
    rows,cols = m.nonzero(as_tuple=True)
    print("  unique rows",rows.unique().numel(),"unique cols",cols.unique().numel(), "max per row", torch.bincount(rows).max().item(), "max per col", torch.bincount(cols).max().item())
    # absolute value quantiles normal entries
    print("  normal |v|/std quantiles:", [round(x,2) for x in (v[~m].abs()/s)[:1000000].quantile(torch.tensor([.5,.9,.99,.999,1.])).tolist()])
