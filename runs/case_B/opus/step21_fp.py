import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
torch.manual_seed(1)
R=torch.load("/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/r5g.pt")
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def ppl(sd): return perplexity(sd,ids,max_tokens=2049)
A=dict(R)
for k,v in R.items():
    if v.dim()!=2: continue
    z,mu,s=detect2(D[k])
    m=torch.rand_like(v)<0.003; o=v.clone(); o[m]=mu[m]; A[k]=o
print("R + 0.3% random entries set to mu:", ppl(A))
# undetectable-type noise only in wte vs only in blocks
for name,sel in [("wte only",lambda k:k=="wte.weight"),("blocks only",lambda k:k.startswith("h."))]:
    B=dict(R)
    for k,v in R.items():
        if v.dim()!=2 or not sel(k): continue
        z,mu,s=detect2(D[k]); m=torch.rand_like(v)<0.0035; o=v.clone(); o[m]+=torch.randn(m.sum())*3*s[m]; B[k]=o
    print("R + small noise,",name, ppl(B))
