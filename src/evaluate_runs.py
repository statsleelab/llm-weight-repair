"""Step 3: score every repaired checkpoint against the held-out suite and
against the (until now hidden) clean weights.

    python evaluate_runs.py <assets> <secret_dir> <runs_dir> <stage_dir> <out.json> [--cases A B ...]

Function-space metrics (held-out benchmarks) come from benchmarks.Suite.
`recovery` = (NLL_damaged - NLL_repaired) / (NLL_damaged - NLL_clean), i.e. the
share of the damage-induced loss gap that the repair closed (1 = fully
recovered, 0 = no better than damaged, <0 = made it worse).

Weight-space metrics, computed with the secret masks:
  residual      ||repaired - clean|| / ||damaged - clean||  on damaged entries
                (1 = untouched, 0 = perfectly restored, >1 = made worse)
  recall        share of damaged entries the repair modified
  precision     share of modified entries that were actually damaged
  collateral    RMS change on *undamaged* entries / RMS of those entries
"""
import argparse, json, math, sys, time
from pathlib import Path
import torch
import gpt2_lib as G
from benchmarks import Suite
from heuristic_baseline import repair as heuristic_repair

CASES = {"A": "scatter_zero", "B": "scatter_noise", "C": "block_zero", "D": "block_noise"}
MODELS = ["fable", "opus", "sonnet", "haiku"]


def weight_metrics(rep, dmg, clean, masks):
    num = den = 0.0
    hit = changed = n_mask = 0
    coll_num = coll_den = 0.0
    per = {}
    for k, m in masks.items():
        r, d, c = rep[k], dmg[k], clean[k]
        e_r, e_d = ((r - c)[m] ** 2).sum().item(), ((d - c)[m] ** 2).sum().item()
        num += e_r; den += e_d
        ch = r != d
        hit += (ch & m).sum().item(); changed += ch.sum().item(); n_mask += m.sum().item()
        coll_num += ((r - c)[~m] ** 2).sum().item(); coll_den += (c[~m] ** 2).sum().item()
        grp = "wte" if k.startswith("wte") else "wpe" if k.startswith("wpe") else k.split(".", 2)[2].rsplit(".", 1)[0]
        g = per.setdefault(grp, [0.0, 0.0]); g[0] += e_r; g[1] += e_d
    bad = sum(int((rep[k] != clean[k]).sum()) for k in clean if k not in masks)
    return {
        "residual": math.sqrt(num / den),
        "recall": hit / n_mask,
        "precision": hit / changed if changed else float("nan"),
        "n_changed": changed,
        "collateral": math.sqrt(coll_num / coll_den),
        "non_matrix_entries_changed": bad,
        "residual_by_group": {g: math.sqrt(a / b) for g, (a, b) in per.items()},
    }


def colmean_zero(dmg, masks=None):
    """POST-HOC blind baseline: only exact zeros are rewritten, with the mean of
    the non-zero entries of their column.  Nothing else is touched."""
    out = {}
    for k, w in dmg.items():
        if w.ndim != 2:
            out[k] = w; continue
        z = w == 0
        keep = (~z).float()
        mean = (w * keep).sum(0, keepdim=True) / keep.sum(0, keepdim=True).clamp(min=1)
        out[k] = torch.where(z, mean.expand_as(w), w)
    return out


def oracle_colmean(dmg, masks):
    """NON-BLIND reference: the secret mask says where the damage is; damaged
    entries are replaced by the mean of the undamaged entries of their column.
    Separates 'finding the damage' from 'guessing the lost values'."""
    out = dict(dmg)
    for k, m in masks.items():
        w = dmg[k]
        keep = (~m).float()
        mean = (w * keep).sum(0, keepdim=True) / keep.sum(0, keepdim=True).clamp(min=1)
        out[k] = torch.where(m, mean.expand_as(w), w)
    return out


SCRIPTED = {"heuristic": lambda d, m: heuristic_repair(d), "colmean_zero": colmean_zero, "oracle_colmean": oracle_colmean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("assets"); ap.add_argument("secret"); ap.add_argument("runs"); ap.add_argument("stage"); ap.add_argument("out")
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    a = ap.parse_args()
    secret, runs, stage, out = Path(a.secret), Path(a.runs), Path(a.stage), Path(a.out)
    suite = Suite(a.assets)
    clean = G.load_weights(secret / "clean.safetensors")
    base = json.loads((secret / "baseline_metrics.json").read_text())
    res = json.loads(out.read_text()) if out.exists() else {"clean": base["clean"], "cases": {}}
    nll_c = base["clean"]["wikitext2_nll"]
    for case in a.cases:
        cond = CASES[case]
        dmg = G.load_weights(stage / f"case_{case}.safetensors")
        masks = torch.load(secret / f"masks_{cond}.pt")
        entry = res["cases"].setdefault(case, {"condition": cond, "damaged": base[cond], "repairs": {}})
        nll_d = base[cond]["wikitext2_nll"]
        cands = {n: None for n in SCRIPTED} | {m: runs / f"case_{case}" / m / "repaired.safetensors" for m in MODELS}
        for name, p in cands.items():
            if name in entry["repairs"]:
                continue
            t = time.time()
            try:
                if p is None:
                    rep = SCRIPTED[name](dmg, masks)
                elif not p.exists():
                    entry["repairs"][name] = {"status": "no_deliverable"}; continue
                else:
                    rep = G.load_weights(p)
                    assert set(rep) == set(clean) and all(rep[k].shape == clean[k].shape for k in clean), "key/shape mismatch"
                r = suite.run(rep) | weight_metrics(rep, dmg, clean, masks)
                r["recovery"] = (nll_d - r["wikitext2_nll"]) / (nll_d - nll_c)
                r["status"] = "ok"
            except Exception as e:  # noqa: BLE001
                r = {"status": f"invalid: {e}"}
            entry["repairs"][name] = r
            print(case, name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "residual_by_group"},
                  f"{time.time() - t:.0f}s", flush=True)
            out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
