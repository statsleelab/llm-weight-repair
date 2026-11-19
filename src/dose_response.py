"""Dose-response: damage fraction x condition x scripted repair.

    python dose_response.py <assets> <clean.safetensors> <out.json> [--fracs ...]

For every (condition, fraction) cell the damaged model and each scripted repair
are scored on a reduced held-out suite (8,192 WikiText-2 test tokens and
670 BLiMP pairs) so that the whole grid fits in a few CPU hours.

Blind candidates : zero_ridge, em_ridge, emg_ridge, block_ridge
auto             : the blind candidate (or no repair) with the lowest
                   perplexity on 6,144 dev tokens -- the same selection right
                   the LLM agents had
colmean_zero     : zeros -> column mean (blind, trivial)
oracle_*         : told the damage mask (not blind)
"""
import argparse, json, math, time
from pathlib import Path
import torch
import gpt2_lib as G
from benchmarks import Suite, detok
from damage import CONDITIONS, damage
import strong_baselines as SB

ap = argparse.ArgumentParser()
ap.add_argument("assets"); ap.add_argument("clean"); ap.add_argument("out")
ap.add_argument("--fracs", nargs="*", type=float, default=[0.01, 0.03, 0.10, 0.001, 0.30])
ap.add_argument("--seed", type=int, default=20261118)
a = ap.parse_args()
assets, out = Path(a.assets), Path(a.out)
suite = Suite(assets, wt_tokens=8192, blimp_per=10)
dev = suite.tok.encode(detok((assets / "data/wikitext2_valid.txt").read_text()[:60000]))[:6144]
clean = G.load_weights(a.clean)
res = json.loads(out.read_text()) if out.exists() else {"cells": {}}


def test_eval(sd):
    nll, acc = G.nll_windows(sd, suite.wt, window=512)
    ok = 0
    for i in range(0, len(suite.blimp), 64):
        b = suite.blimp[i:i + 64]
        lp = G.sentence_logprob(sd, suite.tok, [x[1] for x in b] + [x[2] for x in b])
        ok += sum(lp[j] > lp[j + len(b)] for j in range(len(b)))
    return {"nll": nll, "top1": acc, "blimp": ok / len(suite.blimp)}


def fp(sd):
    return tuple(round(float(sd[k].double().abs().sum()), 4) for k in ("wte.weight", "h.0.attn.c_attn.weight", "h.5.mlp.c_fc.weight", "h.11.mlp.c_proj.weight")) + \
           (round(sum(float(v.double().sum()) for v in sd.values()), 3),)


def wstats(rep, dmg, masks):
    hit = ch = n = 0; num = den = 0.0
    for k, m in masks.items():
        c = rep[k] != dmg[k]
        hit += (c & m).sum().item(); ch += c.sum().item(); n += m.sum().item()
        num += ((rep[k] - clean[k])[m] ** 2).sum().item(); den += ((dmg[k] - clean[k])[m] ** 2).sum().item()
    return {"recall": hit / n, "precision": hit / ch if ch else None, "residual": math.sqrt(num / den)}


if "clean" not in res:
    res["clean"] = test_eval(clean)
for frac in a.fracs:
    for i, (cond, (pattern, value, _, k)) in enumerate(CONDITIONS.items()):
        key = f"{cond}@{frac}"
        if key in res["cells"]:
            continue
        t0 = time.time()
        dmg, masks = damage(clean, pattern, value, frac, a.seed + i, k)
        cache, cell = {}, {"condition": cond, "frac": frac, "methods": {}}

        def score(name, sd, extra=None):
            f = fp(sd)
            if f not in cache:
                cache[f] = test_eval(sd)
            cell["methods"][name] = dict(cache[f]) | (wstats(sd, dmg, masks) if name != "damaged" else {}) | (extra or {})

        score("damaged", dmg)
        cands = {"none": (dmg, G.nll_windows(dmg, dev)[0])}
        for mode in ("zero_ridge", "em_ridge", "emg_ridge", "block_ridge"):
            rep, info = SB.repair_all(dmg, mode)
            n_flag = sum(v.get("n_flag", 0) for v in info.values())
            score(mode, rep, {"n_flagged": n_flag})
            cands[mode] = (rep, G.nll_windows(rep, dev)[0])
            cell["methods"][mode]["dev_nll"] = cands[mode][1]
        pick = min(cands, key=lambda m: cands[m][1] if math.isfinite(cands[m][1]) else 1e9)
        score("auto", cands[pick][0], {"picked": pick})
        del cands
        score("colmean_zero", SB.repair_all(dmg, "colmean_zero")[0])
        score("oracle_colmean", SB.repair_all(dmg, "oracle_colmean", masks)[0])
        score("oracle_ridge", SB.repair_all(dmg, "oracle_ridge", masks)[0])
        cell["seconds"] = time.time() - t0
        res["cells"][key] = cell
        out.write_text(json.dumps(res, indent=1))
        print(key, f"{cell['seconds']:.0f}s", {m: round(math.exp(min(v["nll"], 30)), 1) for m, v in cell["methods"].items()}, "pick", pick, flush=True)
