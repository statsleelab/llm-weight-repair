# Dev perplexity (512-token non-overlapping windows, first 8000 tokens) for damaged vs repaired checkpoints.
import sys, math
sys.path.insert(0, "/home/user/runs/case_C/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, perplexity, nll_windows, Tokenizer
tok = Tokenizer("/home/user/runs/case_C/haiku/tokenizer")
text = open("/home/user/runs/case_C/haiku/dev.txt", encoding="utf-8").read()
ids = tok.encode(text)
print("dev tokens:", len(ids))
for name, path in [("damaged", "/home/user/runs/case_C/haiku/damaged.safetensors"),
                   ("repaired", "/home/user/runs/case_C/haiku/repaired.safetensors")]:
    sd = load_weights(path)
    nll, acc = nll_windows(sd, ids, window=512, max_tokens=8000)
    print(f"{name:9s} dev NLL={nll:.4f}  PPL={math.exp(nll):.3f}  top1={acc:.4f}", flush=True)
