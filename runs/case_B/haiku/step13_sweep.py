import torch, math, sys
torch.set_num_threads(1)
from gpt2_lib import load_weights, forward, Tokenizer, nll_windows
from step07_repair import detect_and_fill
sd0 = load_weights("damaged.safetensors")
tok = Tokenizer("tokenizer")
ids = tok.encode(open("dev.txt").read())
def build(Tlin, Twte, Twpe=None):
    sd = dict(sd0)
    for k in sd0:
        if k.startswith("h.") and k.endswith(("c_attn.weight","c_proj.weight","c_fc.weight")) and Tlin:
            sd[k] = detect_and_fill(sd0[k], Tlin)[0]
    if Twte: sd["wte.weight"] = detect_and_fill(sd0["wte.weight"], Twte)[0]
    if Twpe: sd["wpe.weight"] = detect_and_fill(sd0["wpe.weight"], Twpe)[0]
    return sd
configs = []
for Tlin,Twte,Twpe in configs:
    sd = build(Tlin,Twte,Twpe)
    nll, acc = nll_windows(sd, ids[:4096])
    print(f"Tlin={Tlin} Twte={Twte} Twpe={Twpe} nll4096={nll:.3f} ppl={math.exp(nll):.2f} top1={acc:.4f}", flush=True)
