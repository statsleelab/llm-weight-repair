"""Dev perplexity (512-token windows, first 8000 tokens) for a checkpoint."""
import sys
import torch
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_A/haiku")
from gpt2_lib import load_weights, perplexity, Tokenizer, nll_windows
tok = Tokenizer("/home/user/runs/case_A/haiku/tokenizer")
ids = tok.encode(open("/home/user/runs/case_A/haiku/dev.txt", encoding="utf-8").read())
print("dev tokens:", len(ids))
for path in sys.argv[1:]:
    sd = load_weights(path)
    nll, acc = nll_windows(sd, ids, window=512, max_tokens=8000)
    import math
    print(f"{path}: ppl={math.exp(nll):.3f} nll={nll:.4f} top1={acc:.4f}", flush=True)
