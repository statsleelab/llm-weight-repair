import torch, sys
torch.set_num_threads(1)
sys.path.insert(0, ".")
from gpt2_lib import load_weights, forward, Tokenizer
sd = load_weights("damaged.safetensors")
sd["wpe.weight"] = torch.load("cache/wpe_fixed.pt")["W"]
tok = Tokenizer("tokenizer")
ids = torch.tensor([tok.encode(open("dev.txt", encoding="utf-8").read()[:600])[:128]])
logits, hid = forward(sd, ids, return_hidden=True)
for i, h in enumerate(hid):
    n = h[0].norm(dim=-1)
    print(f"resid {i:2d} norm median={n.median():9.2f} max={n.max():10.2f} absmax={h.abs().max():10.2f}")
print("logits absmax", logits.abs().max().item(), "logit std", logits.std().item())
# which weight rows/cols are dominant in a layer's MLP output
for i in range(12):
    p=f"h.{i}."
    print(i, "c_proj(mlp) absmax", sd[p+"mlp.c_proj.weight"].abs().max().item(), "attn c_proj absmax", sd[p+"attn.c_proj.weight"].abs().max().item(), "c_fc absmax", sd[p+"mlp.c_fc.weight"].abs().max().item(), "c_attn absmax", sd[p+"attn.c_attn.weight"].abs().max().item())
print("wte absmax", sd["wte.weight"].abs().max().item(), "ln_f w mean", sd["ln_f.weight"].mean().item())
