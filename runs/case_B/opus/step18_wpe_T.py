import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
cache={k:detect2(v) for k,v in D.items() if v.dim()==2 and k!="wpe.weight"}
def rep(k,T,pk=5.0):
    v=D[k]; z,mu,s=cache[k]; r=v-mu; tau=r[z.abs()>15].std().item()
    m=(z.abs()>T)&(r.abs()<pk*tau); o=v.clone(); o[m]=mu[m]; return o
def ppl(sd): return perplexity(sd,ids,max_tokens=2049)
W=D["wpe.weight"]
wA,_=repair_wpe2(W)
zw,muw,sw=detect2(W); wB=W.clone(); mB=zw.abs()>5; wB[mB]=muw[mB]
print("wpe generic flagged",mB.sum().item())
for T in [5,3]:
    base={k:rep(k,T) if (v.dim()==2 and k!="wpe.weight") else v for k,v in D.items()}
    for name,wp in [("nbr",wA),("generic",wB)]:
        sd=dict(base); sd["wpe.weight"]=wp
        print(f"T={T} wpe={name} ppl={ppl(sd):.2f}",flush=True)
