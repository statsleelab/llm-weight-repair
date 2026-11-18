import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def ppl(sd): return perplexity(sd,ids,max_tokens=4097)
cache=torch.load("/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/cache2.pt")
def build(nu,pi,pk):
    sd=dict(D); nprot=0
    for k,(z,mu,s) in cache.items():
        w=D[k]; _,p,tau=soft_repair(w,z,mu,s,nu=nu,pi=pi)
        prot=w.abs()>pk*tau; nprot+=(prot&(p>0.5)).sum().item(); p=p.masked_fill(prot,0.0)
        sd[k]=mu+(1-p)*(w-mu)
    return sd,nprot
best=None
for nu,pi,pk in [(4,0.01,3.5),(4,0.03,3.5),(4,0.01,3.0)]:
    sd,npr=build(nu,pi,pk); p=ppl(sd); print(f"nu={nu} pi={pi} pk={pk} protected={npr} ppl4096={p:.2f}",flush=True)
    if p<4524 and (best is None or p<best[0]): best=(p,nu,pi,pk,sd)
if best: print("saving",best[:4]); save_weights(best[4],"repaired.safetensors")
