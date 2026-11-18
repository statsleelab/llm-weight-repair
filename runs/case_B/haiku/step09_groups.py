import torch, math
torch.set_num_threads(1)
from gpt2_lib import load_weights, forward, Tokenizer, nll_windows
from step07_repair import detect_and_fill
sd0 = load_weights("damaged.safetensors")
tok = Tokenizer("tokenizer")
ids = tok.encode(open("dev.txt").read())
x = torch.tensor([[50256]+ids[:128]])
groups = {"none": [], "c_attn": ["attn.c_attn.weight"], "c_proj_attn": ["attn.c_proj.weight"], "c_fc": ["mlp.c_fc.weight"], "mlp_proj": ["mlp.c_proj.weight"], "wte": ["wte.weight"]}
for g, pats in groups.items():
    sd = dict(sd0)
    for k in sd0:
        if any(k.endswith(p) for p in pats) or (g=="none" and False):
            sd[k] = detect_and_fill(sd0[k], 6.0)[0]
    _, hidden = forward(sd, x, return_hidden=True)
    print(f"{g:14s} resid1 mean norm {hidden[1][0].norm(dim=-1).mean().item():8.2f}  resid12 {hidden[12][0].norm(dim=-1).mean().item():9.2f}")
