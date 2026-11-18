"""Write repaired.safetensors (no candidate beat the damaged weights on dev, so
the output keeps the damaged tensors, re-saved as float32) and verify it."""
import torch, sys
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_D/haiku")
from gpt2_lib import load_weights, save_weights, load_weights as lw, Tokenizer, perplexity
from safetensors.torch import load_file
tok = Tokenizer("tokenizer")
ids = tok.encode(open("dev.txt", encoding="utf-8").read())
sd = load_weights("damaged.safetensors")
save_weights(sd, "repaired.safetensors")
rep = load_weights("repaired.safetensors")
src = load_file("damaged.safetensors")
assert set(rep) == set(src), "key mismatch"
assert all(rep[k].shape == src[k].shape for k in src), "shape mismatch"
assert all(rep[k].dtype == torch.float32 for k in rep)
print("keys/shapes/dtype OK:", len(rep), "tensors")
print("max abs diff repaired vs damaged:", max(float((rep[k]-src[k].float()).abs().max()) for k in src))
print("full dev tokens", len(ids))
print("dev ppl damaged  (full dev, window 512):", round(perplexity(sd, ids, window=512), 3), flush=True)
print("dev ppl repaired (full dev, window 512):", round(perplexity(rep, ids, window=512), 3), flush=True)
