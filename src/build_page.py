"""Build docs/index.html and docs/style.css (GitHub Pages) from page/page.html, page/style.css and results.

    python src/build_page.py --author "Name" --affiliation "Affiliation"
"""
import argparse, json, math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--author", default="Donghyung Lee")
ap.add_argument("--affiliation", default="Miami University")
a = ap.parse_args()
root = Path(__file__).resolve().parent.parent


def clean(o):  # NaN/inf are not valid JSON for the browser
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


data = json.loads((root / "results/results.json").read_text())
dose = json.loads((root / "results/dose_response.json").read_text())
keep = ("damaged", "auto", "oracle_ridge")
data["dose"] = {"clean": dose["clean"], "cells": {k: {"methods": {m: {"nll": v["methods"][m]["nll"]} for m in keep}} for k, v in dose["cells"].items()}}
data["lite"] = json.loads((root / "results/llm_lite.json").read_text())
data = clean(data)
frag = (root / "page/page.html").read_text()
frag = frag.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
frag = frag.replace("__AUTHOR__", a.author).replace("__AFFILIATION__", a.affiliation)
(root / "docs").mkdir(exist_ok=True)
(root / "docs/index.html").write_text(frag)
(root / "docs/style.css").write_text((root / "page/style.css").read_text())
(root / "docs/.nojekyll").write_text("")
print("wrote docs/index.html and docs/style.css")
