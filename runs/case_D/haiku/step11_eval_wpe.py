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
res = {}
res["damaged"] = ppl(sd)
for K in ["5.0", "6.0"]:
    s2 = dict(sd); s2["wpe.weight"] = torch.from_numpy(np.load(f"wpe_repaired_K{K}.npy"))
    res["wpe_block_K"+K] = ppl(s2)
# absolute-threshold variant: flag |2nd diff| > 0.0237 and interpolate from neighbours
W = load_file("damaged.safetensors")["wpe.weight"].numpy().astype(np.float64)
n = W.shape[0]
P = np.zeros_like(W); P[1:-1] = (W[:-2] + W[2:]) / 2
flag = np.abs(W - P) > 0.0237; flag[0] = flag[-1] = False
new = W.copy()
for i, j in np.argwhere(flag):
    new[i, j] = P[i, j]
s3 = dict(sd); s3["wpe.weight"] = torch.from_numpy(new.astype(np.float32))
res["wpe_absK_0.0237"] = ppl(s3)
print("flagged absK", int(flag.sum()))
for k, v in res.items(): print(f"{k:22s} ppl={v:.3f}")
