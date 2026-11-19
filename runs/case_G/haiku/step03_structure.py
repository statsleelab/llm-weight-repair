import torch
torch.set_num_threads(1)
from safetensors import safe_open
f = safe_open("damaged.safetensors", "pt")
for k in ["h.7.mlp.c_fc.weight", "h.1.mlp.c_proj.weight", "h.0.attn.c_proj.weight", "h.11.attn.c_proj.weight", "h.4.attn.c_attn.weight", "wte.weight"]:
    t = f.get_tensor(k).float()
    med = t.median(); mad = (t-med).abs().median()*1.4826
    z = (t-med)/mad
    m = z.abs() > 5
    r = m.sum(1).float(); c = m.sum(0).float()
    exp = m.float().mean().item()
    print(f"{k}: frac|z|>5={exp:.5f} rows: max={int(r.max())} mean={r.mean():.2f} top_rows={torch.topk(r,5).indices.tolist()} top_counts={[int(x) for x in torch.topk(r,5).values]}")
    print(f"   cols: max={int(c.max())} mean={c.mean():.2f} top_cols={torch.topk(c,5).indices.tolist()} top_counts={[int(x) for x in torch.topk(c,5).values]}")
    vals = t[m].abs()
    print(f"   |v| of outliers: quantiles 0.05/0.5/0.95 = {[round(x,3) for x in torch.quantile(vals, torch.tensor([0.05,0.5,0.95])).tolist()]}  sign+ frac={(t[m]>0).float().mean():.3f}")
