import torch, sys; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def build(T, wpeT):
    sd=dict(D); n=0
    for k,v in D.items():
        if v.dim()!=2: continue
        if k=="wpe.weight": sd[k],m=repair_wpe(v,wpeT)
        else: sd[k],m=repair_generic(v,T)
        n+=m.sum().item()
    return sd,n
for T,wT in [(8,6),(6,6),(5,6),(4,6),(5,4)]:
    sd,n=build(T,wT)
    print(f"T={T} wpeT={wT} flagged={n} ppl={perplexity(sd,ids,max_tokens=8193):.3f}",flush=True)
