import torch, sys, math, time
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
import step16_family as S
sd, ids, N = S.sd, S.ids, S.N
# data-driven hot residual dims: columns of writer matrices where giant entries recur across layers
cnt = torch.zeros(768)
for k, v in sd.items():
    if v.ndim == 2 and k.endswith("c_proj.weight"):
        D = v - v.median(dim=0, keepdim=True).values; nsd = 10 * v.std() / 1.0488
        cnt += (D.abs() > 3 * nsd).sum(0)
HOT = (cnt >= 8).nonzero().flatten()
def detect_v6(k, W, t, margin, plevel=2.5):
    m, cm, _ = detect_v5(W, t, margin=margin, protect=1e9)
    if k.endswith("c_proj.weight"):
        D = W - cm; nsd = 10 * W.std() / 1.0488
        prot = torch.zeros_like(m); prot[:, HOT] = D[:, HOT].abs() > plevel * nsd
        m = m & ~prot
    return m, cm
def build(cfg, masks=None):
    out = {}
    for k, v in sd.items():
        if v.ndim != 2: out[k] = v
        elif k == "wpe.weight": out[k] = S.wpe
        else:
            m, cm = detect_v6(k, v, **cfg[S.fam(k)]); R = v.clone(); R[m] = cm[m]; out[k] = R
            if masks is not None: masks[k] = m
    return out
if __name__ == "__main__":
    print("hot dims", HOT.tolist(), cnt[HOT].tolist())
    for cfg in eval(sys.argv[1]):
        r = build(cfg)
        nll, acc = nll_windows(r, ids, max_tokens=N); print(f"v6 {cfg} ppl {math.exp(nll):.3f} acc {acc:.4f}", flush=True)
        x = torch.tensor([[EOT]+ids[:255]]); lg, hid = forward(r, x, return_hidden=True)
        print("   pos0 norms:", [round(h[0,0].norm().item()) for h in hid])
