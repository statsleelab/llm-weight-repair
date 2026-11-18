import torch, math
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()!=2: continue
    a=v.abs(); med=a.median().item(); std=v.std().item()
    q=torch.quantile(a.flatten()[::max(1,a.numel()//2000000)], torch.tensor([0.5,0.99,0.999,0.9999]))
    exact0=(v==0).float().mean().item()
    nan=(~torch.isfinite(v)).sum().item()
    # bit patterns
    print(f"{k:28s} {tuple(v.shape)} std={std:.4f} max={a.max().item():.3g} q={[round(x,4) for x in q.tolist()]} zero={exact0:.4f} nonfinite={nan} >10std={(a>10*std).float().mean().item():.5f}")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
print("ntok",len(ids))
print("ppl damaged", perplexity(sd, ids, max_tokens=8192))
