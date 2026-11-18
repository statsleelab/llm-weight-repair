import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def ppl(sd): return perplexity(sd,ids,max_tokens=4097)
cache=torch.load("/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/cache2.pt")
best=None
for nu,pi in [(4,0.01),(4,0.03),(5,0.02),(3,0.03)]:
    sd=dict(D)
    for k,(z,mu,s) in cache.items(): sd[k],_,_=soft_repair(D[k],z,mu,s,nu=nu,pi=pi)
    p=ppl(sd); print(f"nu={nu} pi={pi} ppl4096={p:.2f}",flush=True)
    if best is None or p<best[0]: best=(p,nu,pi,sd)
print("best",best[:3])
save_weights(best[3],"repaired.safetensors")
