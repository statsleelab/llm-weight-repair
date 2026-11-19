import sys, time
sys.path.insert(0, "/home/user/runs/case_E/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, perplexity, nll_windows
tok = Tokenizer("/home/user/runs/case_E/haiku/tokenizer")
text = open("/home/user/runs/case_E/haiku/dev.txt", encoding="utf-8").read()
ids = tok.encode(text)
print("dev tokens:", len(ids))
sd = load_weights("/home/user/runs/case_E/haiku/damaged.safetensors")
t = time.time()
nll, acc = nll_windows(sd, ids, window=512, max_tokens=8192)
print(f"damaged: ppl={torch.exp(torch.tensor(nll)).item():.3f} nll={nll:.4f} top1={acc:.4f} ({time.time()-t:.1f}s)")
