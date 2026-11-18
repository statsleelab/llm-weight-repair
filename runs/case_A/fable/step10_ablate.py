import torch, math
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
dm=load_weights("damaged.safetensors"); c=load_weights("cand_v1.safetensors")
groups={"wte":lambda k:k=="wte.weight","wpe":lambda k:k=="wpe.weight","attn":lambda k:".attn." in k and k.endswith("weight"),"mlp":lambda k:".mlp." in k and k.endswith("weight")}
for name,f in groups.items():
    # damaged model with only this group repaired
    sd={k:(c[k] if f(k) else dm[k]) for k in dm}
    a=math.exp(nll_windows(sd,ids,max_tokens=2048)[0])
    # repaired model with this group left damaged
    sd={k:(dm[k] if f(k) else c[k]) for k in dm}
    b=math.exp(nll_windows(sd,ids,max_tokens=2048)[0])
    print(f"{name}: only-this-repaired ppl {a:.4g} ; all-but-this-repaired ppl {b:.4g}",flush=True)
print("all repaired", math.exp(nll_windows(c,ids,max_tokens=2048)[0]))
