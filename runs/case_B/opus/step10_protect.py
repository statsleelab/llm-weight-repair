import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def build(T, pk):
    sd=dict(D); n=0
    for k,v in D.items():
        if v.dim()!=2: continue
        if k=="wpe.weight": sd[k],m=repair_wpe(v,6)
        else: sd[k],m,tau=repair_protect(v,T,pk)
        n+=m.sum().item()
    return sd,n
for T,pk in [(6,5),(6,4),(8,5),(5,5)]:
    sd,n=build(T,pk)
    print(f"T={T} pk={pk} flagged={n} ppl={perplexity(sd,ids,max_tokens=8193):.3f}",flush=True)
print(generate(sd,tok,"The capital of France is",15))
