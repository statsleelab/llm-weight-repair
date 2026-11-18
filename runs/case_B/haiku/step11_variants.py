import torch, math
torch.set_num_threads(1)
from gpt2_lib import load_weights, forward, Tokenizer, nll_windows
from step07_repair import detect_and_fill
sd0 = load_weights("damaged.safetensors")
tok = Tokenizer("tokenizer")
ids = tok.encode(open("dev.txt").read())
x = torch.tensor([[50256]+ids[:128]])
def build(Tlin, Twte=None, Twpe=None):
    sd = dict(sd0)
    for k in sd0:
        if k.endswith(("c_attn.weight","c_proj.weight","c_fc.weight")) and k.startswith("h."):
            sd[k] = detect_and_fill(sd0[k], Tlin)[0]
        elif k=="wte.weight" and Twte: sd[k]=detect_and_fill(sd0[k], Twte)[0]
        elif k=="wpe.weight" and Twpe: sd[k]=detect_and_fill(sd0[k], Twpe)[0]
    return sd
variants = {"lin4":build(4.0), "lin3":build(3.0), "lin2.5":build(2.5), "lin4+wte4":build(4.0,4.0), "lin4+wpe4":build(4.0,None,4.0), "lin4+wte4+wpe4":build(4.0,4.0,4.0)}
for name, sd in variants.items():
    _, hid = forward(sd, x, return_hidden=True)
    nll, acc = nll_windows(sd, ids, max_tokens=2048)
    print(f"{name:16s} resid1 {hid[1][0,1:].norm(dim=-1).mean().item():8.2f} resid12 {hid[12][0,1:].norm(dim=-1).mean().item():9.2f} nll2048 {nll:8.3f} ppl {math.exp(nll):.3e} top1 {acc:.4f}", flush=True)
