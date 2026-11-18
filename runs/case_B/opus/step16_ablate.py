import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
cache={k:detect2(v) for k,v in D.items() if v.dim()==2 and k!="wpe.weight"}
wpe_fixed,_=repair_wpe2(D["wpe.weight"])
def rep(k,T,pk=5.0):
    v=D[k]; z,mu,s=cache[k]; r=v-mu; tau=r[z.abs()>15].std().item()
    m=(z.abs()>T)&(r.abs()<pk*tau); o=v.clone(); o[m]=mu[m]; return o
base={k:(wpe_fixed if k=="wpe.weight" else rep(k,5)) if v.dim()==2 else v for k,v in D.items()}
def ppl(sd): return perplexity(sd,ids,max_tokens=2049)
print("base T=5 (wpe2)", ppl(base), flush=True)
groups={"emb":["wte.weight"]}
for L in range(12): groups[f"L{L}"]=[k for k in D if k.startswith(f"h.{L}.") and D[k].dim()==2]
for g,ks in groups.items():
    sd=dict(base)
    for k in ks: sd[k]=rep(k,3)
    print(g,"T=3:",round(ppl(sd),2),flush=True)
