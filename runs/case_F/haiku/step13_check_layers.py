import torch, sys
torch.set_num_threads(1)
sys.path.insert(0, ".")
from gpt2_lib import load_weights, forward, Tokenizer
sd = load_weights("damaged.safetensors")
sd["wpe.weight"] = torch.load("cache/wpe_fixed.pt")["W"]
TWO_D = [k for k in sd if sd[k].dim()==2 and k!="wpe.weight"]
for k in TWO_D:
    sd[k] = torch.load(f"cache/{k}__t4.5_r96.pt")["W"]
sd["h.0.attn.c_proj.weight"] = torch.load("cache/h.0.attn.c_proj.weight__t3.0_r96.pt")["W"]
tok = Tokenizer("tokenizer")
ids = torch.tensor([tok.encode(open("dev.txt", encoding="utf-8").read()[:600])[:128]])
logits, hid = forward(sd, ids, return_hidden=True)
print(" ".join(f"{i}:{h[0].norm(dim=-1).median().item():.0f}" for i,h in enumerate(hid)))
print("logit absmax", round(logits.abs().max().item(),1), "logit std", round(logits.std().item(),2))
