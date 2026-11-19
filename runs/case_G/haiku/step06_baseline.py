import sys, time
sys.path.insert(0, "/home/user/runs/case_G/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, nll_windows, generate
tok = Tokenizer("tokenizer")
ids = tok.encode(open("dev.txt", encoding="utf-8").read())
print("dev tokens", len(ids))
sd = load_weights("damaged.safetensors")
t0 = time.time()
nll, acc = nll_windows(sd, ids, window=512, max_tokens=8192, batch=4)
print(f"damaged: ppl={torch.exp(torch.tensor(nll)).item():.3f} nll={nll:.4f} top1={acc:.4f} time={time.time()-t0:.1f}s")
print(generate(sd, tok, "The capital of France is", 15))
