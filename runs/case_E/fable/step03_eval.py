import torch, sys, math, time
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
tok = Tokenizer("tokenizer"); ids = tok.encode(open("dev.txt").read()); print("dev tokens", len(ids))
sd = load_weights("damaged.safetensors")
N = 4096
t0=time.time(); print("damaged ppl", perplexity(sd, ids, max_tokens=N), time.time()-t0, flush=True)
for t in [float(a) for a in sys.argv[1:]]:
    r = repair_all(sd, t, verbose=(t==float(sys.argv[1])))
    nll, acc = nll_windows(r, ids, max_tokens=N)
    print(f"t={t} ppl {math.exp(nll):.3f} acc {acc:.4f}", flush=True)
