import sys
sys.path.insert(0, "/home/user/runs/case_E/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, nll_windows
tok = Tokenizer("/home/user/runs/case_E/haiku/tokenizer")
ids = tok.encode(open("/home/user/runs/case_E/haiku/dev.txt", encoding="utf-8").read())[:1024]
base = load_weights("/home/user/runs/case_E/haiku/damaged.safetensors")
def ppl(sd):
    nll, acc = nll_windows(sd, ids, window=512, max_tokens=1024)
    return torch.exp(torch.tensor(nll)).item(), acc
# ablate the attention output projection (weight and bias) of one block
for i in [11, 10, 9, 2, 0]:
    sd = dict(base)
    p = f"h.{i}."
    sd[p+"attn.c_proj.weight"] = torch.zeros_like(base[p+"attn.c_proj.weight"])
    sd[p+"attn.c_proj.bias"] = torch.zeros_like(base[p+"attn.c_proj.bias"])
    a, b = ppl(sd); print(f"zero attn.c_proj of block {i}: ppl={a:.3f} top1={b:.4f}", flush=True)
for i in [11, 10, 2, 3]:
    sd = dict(base)
    p = f"h.{i}."
    sd[p+"mlp.c_proj.weight"] = torch.zeros_like(base[p+"mlp.c_proj.weight"])
    sd[p+"mlp.c_proj.bias"] = torch.zeros_like(base[p+"mlp.c_proj.bias"])
    a, b = ppl(sd); print(f"zero mlp.c_proj of block {i}: ppl={a:.3f} top1={b:.4f}", flush=True)
