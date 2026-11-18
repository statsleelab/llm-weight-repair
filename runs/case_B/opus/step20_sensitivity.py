import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
torch.manual_seed(0)
R=torch.load("/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/r5g.pt")
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def ppl(sd): return perplexity(sd,ids,max_tokens=2049)
print("R", ppl(R))
# (a) extra small noise on 0.35% entries, ~3 sigma_ij (sigma from detect2 scale)
S={}
A=dict(R)
for k,v in R.items():
    if v.dim()!=2: continue
    z,mu,s=detect2(D[k]); S[k]=(mu,s)
    m=torch.rand_like(v)<0.0035
    o=v.clone(); o[m]+=torch.randn(m.sum())*3*s[m]; A[k]=o
print("R + small undetectable-type noise", ppl(A))
# (b) re-repair R with same pipeline (false-positive churn)
B=dict(R)
for k,v in R.items():
    if v.dim()!=2: continue
    mu,s=S[k]; z=(v-mu)/s; m=z.abs()>5; o=v.clone(); o[m]=mu[m]; B[k]=o
    print(k, "re-flag", m.sum().item()) if k in ("wte.weight","h.0.mlp.c_fc.weight") else None
print("R re-repaired", ppl(B))
