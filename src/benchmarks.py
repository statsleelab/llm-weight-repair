"""Held-out benchmarks.  Nothing in this file (or the data it reads) is ever
visible to a repair agent.

1. WikiText-2 *test* perplexity + next-token top-1 accuracy
2. BLiMP (Warstadt et al. 2020): 67 paradigms of grammatical minimal pairs
3. Capitals: 'The capital of X is' -- 80-way forced choice factual recall
4. LM consistency with the reference model is NOT a benchmark; weight-space
   distances are reported separately in analyze.py
"""
from __future__ import annotations
import json, math, random
from pathlib import Path
import torch
import gpt2_lib as G

CAPITALS = {
 "France":"Paris","Germany":"Berlin","Italy":"Rome","Spain":"Madrid","Portugal":"Lisbon","Russia":"Moscow",
 "Japan":"Tokyo","China":"Beijing","India":"Delhi","Egypt":"Cairo","Greece":"Athens","Turkey":"Ankara",
 "Iran":"Tehran","Iraq":"Baghdad","Syria":"Damascus","Lebanon":"Beirut","Poland":"Warsaw","Austria":"Vienna",
 "Hungary":"Budapest","Ireland":"Dublin","Norway":"Oslo","Sweden":"Stockholm","Finland":"Helsinki",
 "Denmark":"Copenhagen","Belgium":"Brussels","Cuba":"Havana","Peru":"Lima","Chile":"Santiago",
 "Argentina":"Buenos","Venezuela":"Caracas","Colombia":"Bogot","Kenya":"Nairobi","Ethiopia":"Addis",
 "Thailand":"Bangkok","Vietnam":"Hanoi","Indonesia":"Jakarta","Philippines":"Manila","Afghanistan":"Kabul",
 "Pakistan":"Islamabad","Bangladesh":"Dhaka","Nepal":"Kathmandu","Ukraine":"Kiev","Romania":"Bucharest",
 "Bulgaria":"Sofia","Serbia":"Belgrade","Croatia":"Zagreb","Canada":"Ottawa","Australia":"Canberra",
 "England":"London","Scotland":"Edinburgh","Taiwan":"Taipei","Mongolia":"Ulaan","Libya":"Tripoli",
 "Sudan":"Khartoum","Somalia":"Mogadishu","Uganda":"Kampala","Ghana":"Accra","Senegal":"Dakar",
 "Zimbabwe":"Harare","Jordan":"Amman","Qatar":"Doha","Yemen":"Sana","Malaysia":"Kuala","Cambodia":"Phnom",
 "Iceland":"Reykjavik","Switzerland":"Bern","Netherlands":"Amsterdam","Belarus":"Minsk","Latvia":"Riga",
 "Estonia":"Tallinn","Lithuania":"Vilnius","Georgia":"Tbilisi","Armenia":"Yerevan","Azerbaijan":"Baku",
 "Kazakhstan":"Astana","Uzbekistan":"Tashkent","Haiti":"Port","Jamaica":"Kingston","Uruguay":"Montevideo",
 "Ecuador":"Quito",
}


def detok(t: str) -> str:
    """Undo the PTB-style tokenisation of the pytorch/examples WikiText-2 dump."""
    for a, b in [(" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" , ", ", "), (" . ", ". "), (" ; ", "; "),
                 (" : ", ": "), (" 's", "'s"), (" n't", "n't"), ("( ", "("), (" )", ")"), (" '", "'")]:
        t = t.replace(a, b)
    return t


class Suite:
    def __init__(self, assets: str | Path, wt_tokens=40_000, blimp_per=50, seed=0):
        a = Path(assets)
        self.tok = G.Tokenizer(a / "gpt2")
        self.wt = self.tok.encode(detok((a / "data/wikitext2_test.txt").read_text()))[:wt_tokens]
        rng = random.Random(seed)
        self.blimp = []
        for f in sorted((a / "data/blimp_repo/data").glob("*.jsonl")):
            rows = [json.loads(l) for l in f.read_text().splitlines()]
            for r in rng.sample(rows, blimp_per):
                self.blimp.append((f.stem, r["sentence_good"], r["sentence_bad"]))
        self.cap_first = {c: self.tok.encode(" " + cap)[0] for c, cap in CAPITALS.items()}

    def run(self, sd) -> dict:
        nll, acc = G.nll_windows(sd, self.wt, window=512)
        out = {"wikitext2_nll": nll, "wikitext2_ppl": math.exp(min(nll, 50)), "wikitext2_top1": acc}
        ok = 0
        for i in range(0, len(self.blimp), 64):
            b = self.blimp[i:i + 64]
            lp = G.sentence_logprob(sd, self.tok, [x[1] for x in b] + [x[2] for x in b])
            ok += sum(lp[j] > lp[j + len(b)] for j in range(len(b)))
        out["blimp_acc"] = ok / len(self.blimp)
        cand = torch.tensor(sorted(set(self.cap_first.values())))
        hit = 0
        for c, t in self.cap_first.items():
            lg = G.forward(sd, torch.tensor([[G.EOT] + self.tok.encode(f"The capital of {c} is")]))[0, -1]
            hit += int(cand[lg[cand].argmax()]) == t
        out["capitals_acc"] = hit / len(self.cap_first)
        return out
