"""Build a repaired checkpoint: every exact-zero entry of a 2-D weight is treated as erased and
re-estimated by rank-r hard-impute completion; observed entries are kept verbatim.
Ranks per family were chosen from held-out validation (step04/step05 logs).
usage: step07_repair.py OUT.safetensors [wte_rank]"""
import sys, time
import torch
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_A/haiku")
from gpt2_repair_lib import hard_impute
from safetensors.torch import load_file, save_file

out = sys.argv[1]
wte_rank = int(sys.argv[2]) if len(sys.argv) > 2 else 8
src = load_file("/home/user/runs/case_A/haiku/damaged.safetensors")

def rank_for(name):
    if name == "wpe.weight": return 16
    if name == "wte.weight": return wte_rank
    if name.endswith("attn.c_attn.weight"): return 64
    if name.endswith("attn.c_proj.weight"): return 8
    if name.endswith("mlp.c_fc.weight"): return 8
    return None  # mlp.c_proj and anything else: leave erased entries at zero

rep = {}
log = []
for k, v in src.items():
    v = v.float().clone()
    r = rank_for(k) if v.dim() == 2 else None
    if r is not None:
        miss = (v == 0)
        t0 = time.time()
        X, its = hard_impute(v, miss, r, iters=40)
        n = int(miss.sum())
        log.append(f"{k}: rank={r} filled={n} iters={its} t={time.time()-t0:.1f}s")
        print(log[-1], flush=True)
        v = X.contiguous()
    rep[k] = v.contiguous()
save_file(rep, out)
print("saved", out, "tensors", len(rep))
