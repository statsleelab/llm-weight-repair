"""Which damaged blocks matter? Evaluate ppl (4096 tokens) when ridge-repairing subsets."""
import torch
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
D=load_weights("damaged.safetensors"); Rp=load_weights("cand_ridge.safetensors")
def ppl(sd): return math.exp(nll_windows(sd,ids,max_tokens=4096)[0])
print("damaged",ppl(D)); print("ridge all",ppl(Rp))
groups={"wte":["wte.weight"],"wpe":["wpe.weight"]}
for i in range(12):
    groups[f"L{i}"]=[k for k in D if k.startswith(f"h.{i}.") and D[k].dim()==2]
for g,ks in groups.items():
    sd=dict(Rp)
    for k in ks: sd[k]=D[k]   # put damage back in only this group
    print(f"ridge-all but damaged {g}: {ppl(sd):.2f}",flush=True)
