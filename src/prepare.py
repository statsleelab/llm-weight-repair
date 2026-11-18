"""Step 1: build the clean reference, the damaged checkpoints, the secret masks,
and evaluate clean + damaged on the held-out suite.

    python prepare.py <assets_dir> <out_dir> [--seed 20261118]
"""
import argparse, json, sys, time
from pathlib import Path
import torch
import gpt2_lib as G
from damage import CONDITIONS, damage
from benchmarks import Suite

ap = argparse.ArgumentParser()
ap.add_argument("assets"); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=20261118)
ap.add_argument("--only", nargs="*")
a = ap.parse_args()
assets, out = Path(a.assets), Path(a.out)
out.mkdir(parents=True, exist_ok=True)

clean_p = out / "clean.safetensors"
if not clean_p.exists():
    raw = torch.load(assets / "gpt2/pytorch_model.bin", map_location="cpu", weights_only=True)
    sd = {k: v.float() for k, v in raw.items() if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")}
    G.save_weights(sd, clean_p)
clean = G.load_weights(clean_p)
n_total = sum(v.numel() for v in clean.values())
suite = Suite(assets)
res_p = out / "baseline_metrics.json"
res = json.loads(res_p.read_text()) if res_p.exists() else {}
if "clean" not in res:
    t = time.time(); res["clean"] = suite.run(clean); print("clean", res["clean"], f"{time.time()-t:.0f}s", flush=True)
for i, (name, (pattern, value, frac, k)) in enumerate(CONDITIONS.items()):
    if a.only and name not in a.only: continue
    dmg, masks = damage(clean, pattern, value, frac, a.seed + i, k)
    G.save_weights(dmg, out / f"damaged_{name}.safetensors")
    torch.save(masks, out / f"masks_{name}.pt")
    n_hit = sum(int(m.sum()) for m in masks.values())
    res[name] = suite.run(dmg) | {"n_damaged": n_hit, "frac_of_model": n_hit / n_total}
    print(name, res[name], flush=True)
    res_p.write_text(json.dumps(res, indent=1))
