import torch, sys
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_D/haiku")
from gpt2_lib import load_weights, Tokenizer, perplexity
from safetensors.torch import load_file
import numpy as np
tok = Tokenizer("tokenizer")
ids = tok.encode(open("dev.txt", encoding="utf-8").read())
sd = load_weights("damaged.safetensors")
def ppl(s): return perplexity(s, ids, window=512, max_tokens=8000)
W = load_file("damaged.safetensors")["wpe.weight"].numpy().astype(np.float64)
P = np.zeros_like(W); P[1:-1] = (W[:-2] + W[2:]) / 2
D = W - P; D[0] = 0; D[-1] = 0
out = {}
for T in [0.1, 0.05]:
    flag = np.abs(D) > T
    new = np.where(flag, P, W)
    s2 = dict(sd); s2["wpe.weight"] = torch.from_numpy(new.astype(np.float32))
    out[T] = (int(flag.sum()), ppl(s2))
    print(f"|d|>{T}: flagged {out[T][0]}  ppl={out[T][1]:.3f}", flush=True)
