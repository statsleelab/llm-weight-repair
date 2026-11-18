import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
torch.manual_seed(0)
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def ppl(sd): return perplexity(sd,ids,max_tokens=1025)
cache=torch.load("/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/cache2.pt")
R=dict(D); taus={}
for k,(z,mu,s) in cache.items(): R[k],p,taus[k]=soft_repair(D[k],z,mu,s,nu=4)
C=dict(R); M={}
for k in cache:
    v=R[k]; m=torch.rand_like(v)<0.01; o=v.clone(); o[m]=torch.randn(m.sum())*taus[k]; C[k]=o; M[k]=m
RR=dict(C); S={}
for k in cache:
    z,mu,s=detect2(C[k]); S[k]=(z,mu,s); RR[k],_,_=soft_repair(C[k],z,mu,s,nu=4)
print("R",ppl(R),"RR",ppl(RR),flush=True)
groups={"wte":lambda k:k=="wte.weight","wpe":lambda k:k=="wpe.weight","L0":lambda k:k.startswith("h.0."),
 "L1":lambda k:k.startswith("h.1."),"c_attn":lambda k:"c_attn" in k,"attn.c_proj":lambda k:"attn.c_proj" in k,
 "c_fc":lambda k:"c_fc" in k,"mlp.c_proj":lambda k:"mlp.c_proj" in k}
for g,f in groups.items():
    sd=dict(RR)
    for k in cache:
        if f(k): sd[k]=R[k]
    print("restore",g,round(ppl(sd),1),flush=True)
# error anatomy for a few matrices
for k in ["h.5.mlp.c_fc.weight","h.5.attn.c_attn.weight","wte.weight"]:
    e=(RR[k]-R[k]); m=M[k]
    print(k,"err^2 on corrupted %.3f on clean %.3f"%((e[m]**2).sum(),(e[~m]**2).sum()),
          "missed(|e|>3s) %d"%((e.abs()>3*S[k][2])&m).sum().item())
