"""Build docs/index.html (GitHub Pages) and build/artifact.html from page/page.html + results.

    python src/build_page.py --author "Name" --affiliation "Affiliation"
"""
import argparse, json, math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--author", default="StatsLeeLab")
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


data = clean(json.loads((root / "results/results.json").read_text()))
frag = (root / "page/page.html").read_text()
frag = frag.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
frag = frag.replace("__AUTHOR__", a.author).replace("__AFFILIATION__", a.affiliation)
(root / "build").mkdir(exist_ok=True)
(root / "build/artifact.html").write_text(frag)
head, body = frag.split("</style>", 1)
doc = ('<!doctype html>\n<html lang="en" data-lang="en">\n<head>\n<meta charset="utf-8">\n'
       '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
       + head + "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")
(root / "docs").mkdir(exist_ok=True)
(root / "docs/index.html").write_text(doc)
(root / "docs/.nojekyll").write_text("")
print("wrote docs/index.html and build/artifact.html")
