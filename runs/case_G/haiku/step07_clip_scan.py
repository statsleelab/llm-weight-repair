import sys, time
sys.path.insert(0, "/home/user/runs/case_G/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, nll_windows
tok = Tokenizer("tokenizer")
ids = tok.encode(open("dev.txt", encoding="utf-8").read())
sd0 = load_weights("damaged.safetensors")
MAT = [k for k in sd0 if sd0[k].dim() == 2 and k not in ("wte.weight", "wpe.weight")]
def ppl(sd):
    nll, _ = nll_windows(sd, ids, window=512, max_tokens=4096, batch=4)
    return float(torch.exp(torch.tensor(nll)))
print("baseline 4096 tok:", round(ppl(sd0), 2), flush=True)
for T in [float(x) for x in sys.argv[1:]]:
    t0 = time.time()
    sd = dict(sd0)
    nz = 0
    for k in MAT:
        w = sd0[k]
        med = w.median(); mad = (w - med).abs().median() * 1.4826
        m = (w - med).abs() > T * mad
        nz += int(m.sum())
        sd[k] = torch.where(m, torch.zeros_like(w), w)
    print(f"T={T} zeroed={nz} ({nz/sum(sd0[k].numel() for k in MAT):.4f}) ppl={ppl(sd):.2f} time={time.time()-t0:.1f}", flush=True)
