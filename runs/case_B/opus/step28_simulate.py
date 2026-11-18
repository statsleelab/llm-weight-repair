import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
torch.manual_seed(0)
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def ppl(sd): return perplexity(sd,ids,max_tokens=2049)
cache=torch.load("/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/cache2.pt")
R=dict(D); taus={}
for k,(z,mu,s) in cache.items(): R[k],p,taus[k]=soft_repair(D[k],z,mu,s,nu=4)
print("R (pseudo-clean) ppl", ppl(R), flush=True)
C=dict(R)
for k in cache:
    v=R[k]; m=torch.rand_like(v)<0.01; o=v.clone(); o[m]=torch.randn(m.sum())*taus[k]; C[k]=o
print("R corrupted ppl", ppl(C), flush=True)
RR=dict(C)
for k in cache:
    z,mu,s=detect2(C[k]); RR[k],_,_=soft_repair(C[k],z,mu,s,nu=4)
print("R corrupted+repaired ppl", ppl(RR), flush=True)
