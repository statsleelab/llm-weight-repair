"""Score LLM repairs on the lite suite used by dose_response.py so they sit on the dose curve.
    python score_llm_lite.py <assets> <secret_dir> <runs_dir> <out.json>
Cases A-D are the 1% conditions; E-G come from make_cells.py."""
import json, math, sys
from pathlib import Path
import torch
import gpt2_lib as G
from benchmarks import Suite
from damage import CONDITIONS, damage
from make_cells import CELLS, SEED

ALL = {"A": ("scatter_zero", 0.01), "B": ("scatter_noise", 0.01), "C": ("block_zero", 0.01), "D": ("block_noise", 0.01)} | CELLS
MODELS = ["fable", "opus", "sonnet", "haiku"]
assets, secret, runs, out = map(Path, sys.argv[1:5])
torch.set_num_threads(4)
suite = Suite(assets, wt_tokens=8192, blimp_per=10)
clean = G.load_weights(secret / "clean.safetensors")
res = json.loads(out.read_text()) if out.exists() else {}


def test_eval(sd):
    nll, acc = G.nll_windows(sd, suite.wt, window=512)
    ok = 0
    for i in range(0, len(suite.blimp), 64):
        b = suite.blimp[i:i + 64]
        lp = G.sentence_logprob(sd, suite.tok, [x[1] for x in b] + [x[2] for x in b])
        ok += sum(lp[j] > lp[j + len(b)] for j in range(len(b)))
    return {"nll": nll, "top1": acc, "blimp": ok / len(suite.blimp)}


for case, (cond, frac) in ALL.items():
    todo = [m for m in MODELS if f"{case}/{m}" not in res]
    if not todo:
        continue
    pattern, value, _, k = CONDITIONS[cond]
    dmg, masks = damage(clean, pattern, value, frac, SEED + list(CONDITIONS).index(cond), k)
    for m in todo:
        p = runs / f"case_{case}" / m / "repaired.safetensors"
        if not p.exists():
            res[f"{case}/{m}"] = {"status": "no_deliverable"}; continue
        rep = G.load_weights(p)
        r = test_eval(rep)
        hit = ch = n = 0; num = den = 0.0
        for kk, mk in masks.items():
            c = rep[kk] != dmg[kk]
            hit += (c & mk).sum().item(); ch += c.sum().item(); n += mk.sum().item()
            num += ((rep[kk] - clean[kk])[mk] ** 2).sum().item(); den += ((dmg[kk] - clean[kk])[mk] ** 2).sum().item()
        other = sum(int((rep[kk] != clean[kk]).sum()) for kk in clean if kk not in masks)
        r |= {"condition": cond, "frac": frac, "recall": hit / n, "precision": hit / ch if ch else None,
              "residual": math.sqrt(num / den), "n_changed": ch, "non_matrix_changed": other}
        res[f"{case}/{m}"] = r
        out.write_text(json.dumps(res, indent=1))
        print(case, m, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
