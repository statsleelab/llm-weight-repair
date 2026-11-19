# Assemble a candidate repair and measure dev perplexity (diagnostic selection only).
# usage: python step10_eval.py MODE [SAVE]   MODE in {none, wpe, wpe+zero, wpe+lr}
import torch, sys, time, math
torch.set_num_threads(1)
sys.path.insert(0, ".")
from gpt2_lib import load_weights, save_weights, Tokenizer, nll_windows
from safetensors.torch import load_file
mode = sys.argv[1]; save = len(sys.argv) > 2
sd = load_weights("damaged.safetensors")
TWO_D = [k for k in sd if sd[k].dim() == 2 and k != "wpe.weight"]
if mode != "none":
    sd["wpe.weight"] = torch.load("cache/wpe_fixed.pt")["W"]
if mode in ("wpe+zero", "wpe+lr", "wpe+zero+fix0", "wpe+lr+fix0"):
    for k in TWO_D:
        c = torch.load(f"cache/{k}__t4.5_r96.pt")
        if mode.startswith("wpe+zero"):
            sd[k] = torch.where(c["mask"], torch.zeros_like(sd[k]), sd[k])
        else:
            sd[k] = c["W"]
    if mode.endswith("fix0"):
        # block-0 attention projection has a much narrower bulk: use a 3-sigma detector
        c = torch.load("cache/h.0.attn.c_proj.weight__t3.0_r96.pt")
        if mode.startswith("wpe+zero"):
            sd["h.0.attn.c_proj.weight"] = torch.where(c["mask"], torch.zeros_like(sd["h.0.attn.c_proj.weight"]), sd["h.0.attn.c_proj.weight"])
        else:
            sd["h.0.attn.c_proj.weight"] = c["W"]
if save:
    save_weights(sd, "repaired.safetensors"); print("saved repaired.safetensors")
if len(sys.argv) > 3 or not save:
    pass
tok = Tokenizer("tokenizer")
ids = tok.encode(open("dev.txt", encoding="utf-8").read())
t0 = time.time()
nll, acc = nll_windows(sd, ids, window=512, max_tokens=8192, batch=4)
print(f"mode={mode} ntok=8192 dev_ppl={math.exp(nll):.3f} top1={acc:.4f} t={time.time()-t0:.1f}s", flush=True)
