import sys
sys.path.insert(0, "/home/user/runs/case_E/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, nll_windows
tok = Tokenizer("/home/user/runs/case_E/haiku/tokenizer")
ids = tok.encode(open("/home/user/runs/case_E/haiku/dev.txt", encoding="utf-8").read())[:1024]
base = load_weights("/home/user/runs/case_E/haiku/damaged.safetensors")
def ev(name, sd):
    nll, acc = nll_windows(sd, ids, window=512, max_tokens=1024)
    print(f"{name:55s} nll={nll:9.4f} ppl={min(torch.exp(torch.tensor(min(nll,80.0))).item(),1e35):12.3f} top1={acc:.4f}", flush=True)
def zc(sd, k, cols=None, rows=None):
    W = sd[k].clone()
    if cols is not None: W[:, cols] = 0
    if rows is not None: W[rows, :] = 0
    sd[k] = W
L = range(12)
sd = dict(base)
for i in L:
    zc(sd, f"h.{i}.mlp.c_proj.weight", cols=[138, 447])
ev("zero mlp.c_proj cols 138,447 all layers", sd)
sd = dict(base)
for i in L:
    zc(sd, f"h.{i}.mlp.c_proj.weight", cols=[138, 447]); zc(sd, f"h.{i}.attn.c_proj.weight", cols=[138, 447])
ev("zero attn+mlp c_proj cols 138,447 all layers", sd)
sd = dict(base); zc(sd, "h.2.mlp.c_proj.weight", rows=[3034])
ev("zero h.2 mlp.c_proj row 3034", sd)
sd = dict(base); zc(sd, "h.2.mlp.c_fc.weight", cols=[3034])
ev("zero h.2 mlp.c_fc col 3034", sd)
sd = dict(base)
for i in L:
    k = f"h.{i}.mlp.c_proj.weight"; W = sd[k].clone(); W[W.abs() > 5] = 0; sd[k] = W
ev("zero |w|>5 in all mlp.c_proj", sd)
sd = dict(base); 
for i in L:
    for n in ["attn.c_proj", "mlp.c_proj"]:
        k = f"h.{i}.{n}.weight"; W = sd[k].clone(); W[W.abs() > 5] = 0; sd[k] = W
ev("zero |w|>5 in all attn+mlp c_proj", sd)
