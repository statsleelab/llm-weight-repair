import torch, sys, math, time
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
import step16_family as S
sd, ids, N = S.sd, S.ids, S.N
def block_z(W, axis):
    # robust z of each entry within its 64-wide head block along `axis`
    cm = W.median(dim=0, keepdim=True).values
    D = W - cm
    if axis == 1: B = D.view(D.shape[0], -1, 64)
    else: B = D.view(-1, 64, D.shape[1]).transpose(1, 2)          # (heads, cols, 64)
    med = B.median(dim=-1, keepdim=True).values
    s = 1.4826 * (B - med).abs().median(dim=-1, keepdim=True).values + 1e-9
    Zb = (B - med).abs() / s
    if axis == 1: return Zb.reshape(D.shape)
    return Zb.transpose(1, 2).reshape(D.shape)
cacheb = {}
def rep_attn(k, tb, **kw):
    key = (k, tb, tuple(sorted(kw.items())))
    if key not in cacheb:
        v = sd[k]; m, cm, _ = detect_v5(v, **kw)
        Zb = block_z(v, 1 if "c_attn" in k else 0)
        m = m & (Zb > tb); R = v.clone(); R[m] = cm[m]; cacheb[key] = R
    return cacheb[key]
def build(attn_cfg, other):
    out = {}
    for k, v in sd.items():
        if v.ndim != 2: out[k] = v
        elif k == "wpe.weight": out[k] = S.wpe
        elif "attn" in k and attn_cfg is not None: out[k] = rep_attn(k, **attn_cfg)
        else: out[k] = S.rep(k, **other[S.fam(k)])
    return out
if __name__ == "__main__":
    for attn_cfg, other in eval(sys.argv[1]):
        nll, acc = nll_windows(build(attn_cfg, other), ids, max_tokens=N)
        print(f"attn {attn_cfg} other {other} ppl {math.exp(nll):.3f} acc {acc:.4f}", flush=True)
