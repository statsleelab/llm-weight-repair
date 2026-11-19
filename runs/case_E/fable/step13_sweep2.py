import torch, sys, math, time
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
tok = Tokenizer("tokenizer"); ids = tok.encode(open("dev.txt").read())
sd = load_weights("damaged.safetensors")
N = 3072
cfgs = eval(sys.argv[1])
for t, p, tw in cfgs:
    r = repair_all_v4(sd, t, t_wpe=tw, protect=p)
    nll, acc = nll_windows(r, ids, max_tokens=N); print(f"v4 t={t} protect={p} t_wpe={tw} ppl {math.exp(nll):.3f} acc {acc:.4f}", flush=True)
