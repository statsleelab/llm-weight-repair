import sys, time
sys.path.insert(0, "/home/user/runs/case_E/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, nll_windows
tok = Tokenizer("/home/user/runs/case_E/haiku/tokenizer")
ids = tok.encode(open("/home/user/runs/case_E/haiku/dev.txt", encoding="utf-8").read())[:2048]
base = load_weights("/home/user/runs/case_E/haiku/damaged.safetensors")

def robust_sig(W):
    a = W.flatten()
    g = torch.Generator().manual_seed(0)
    s = a[torch.randperm(a.numel(), generator=g)[:200000]]
    q = torch.quantile(s, torch.tensor([0.25, 0.75]))
    return ((q[1] - q[0]) / 1.349).item()

def clip(sd, pred, t):
    out = dict(sd)
    for k, W in sd.items():
        if W.dim() != 2 or not pred(k):
            continue
        s = robust_sig(W)
        W2 = W.clone()
        W2[W2.abs() > t * s] = 0.0
        out[k] = W2
    return out

def ppl(sd):
    nll, acc = nll_windows(sd, ids, window=512, max_tokens=2048)
    return torch.exp(torch.tensor(nll)).item(), acc

def is_(*subs):
    return lambda k: any(s in k for s in subs)

variants = [
    ("none", None),
    ("all_2d_6sig", (is_("c_attn", "c_proj", "c_fc", "wte", "wpe"), 6)),
    ("all_2d_4sig", (is_("c_attn", "c_proj", "c_fc", "wte", "wpe"), 4)),
    ("mlp_c_proj_6sig", (is_("mlp.c_proj"), 6)),
    ("attn_c_proj_6sig", (is_("attn.c_proj"), 6)),
    ("c_attn_6sig", (is_("c_attn"), 6)),
    ("c_fc_6sig", (is_("c_fc"), 6)),
    ("wte_wpe_6sig", (is_("wte", "wpe"), 6)),
]
for name, spec in variants:
    t0 = time.time()
    sd = base if spec is None else clip(base, spec[0], spec[1])
    p, a = ppl(sd)
    print(f"{name:20s} ppl={p:10.3f} top1={a:.4f} ({time.time()-t0:.1f}s)", flush=True)
