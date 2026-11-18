import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def ppl(sd): return perplexity(sd,ids,max_tokens=2049)
cache={k:detect2(v) for k,v in D.items() if v.dim()==2}
torch.save(cache,"/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/cache2.pt")
for nu,hard in [(4,False),(4,True),(2,False),(8,False)]:
    sd=dict(D); taus=[]; ne=0
    for k,(z,mu,s) in cache.items():
        sd[k],p,tau=soft_repair(D[k],z,mu,s,nu=nu,hard=hard); taus.append(round(tau,2)); ne+=p.sum().item()
    print(f"nu={nu} hard={hard} expected#corrupt={ne:.0f} ppl={ppl(sd):.2f}", flush=True)
print("taus", taus)
