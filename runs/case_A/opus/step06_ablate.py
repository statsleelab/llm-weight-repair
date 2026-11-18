import torch, math; torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def ppl(s): return perplexity(s, ids, max_tokens=1024)
print("base", ppl(sd))
for i in range(12):
    for part in ["attn","mlp"]:
        s=dict(sd); p=f"h.{i}.{part}.c_proj."
        s[p+"weight"]=torch.zeros_like(sd[p+"weight"]); s[p+"bias"]=torch.zeros_like(sd[p+"bias"])
        print(i,part, f"{ppl(s):.3g}", flush=True)
