import torch, numpy as np, time
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
print(len(ids))
t=time.time();print("ppl damaged",perplexity(sd,ids,max_tokens=4096),time.time()-t)
print(generate(sd,tok,"The capital of France is",15))
# bit-level
for k in ["h.5.attn.c_attn.weight","h.5.mlp.c_fc.weight","wte.weight","h.0.attn.c_proj.weight","wpe.weight"]:
    w=sd[k].numpy().view(np.uint32).ravel()
    print(k,"low8 zero frac",(w&0xff==0).mean(),"low12 zero",(w&0xfff==0).mean())
    v=sd[k]
    print(" kurt",(((v-v.mean())/v.std())**4).mean().item(), "nonfinite",(~torch.isfinite(v)).sum().item())
# per-row/col stats of wpe
wpe=sd["wpe.weight"]
print("wpe row norms", wpe.norm(dim=1)[:10], wpe.norm(dim=1)[-5:])
print("wpe col std", wpe.std(0)[:10], "col abs max", wpe.abs().max(0).values.topk(5))
