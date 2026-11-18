import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
cache={}
for k,v in D.items():
    if v.dim()!=2 or k=="wpe.weight": continue
    cache[k]=detect2(v)
torch.save(cache,"/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/detect2.pt") if False else None
wpe_fixed,_=repair_wpe(D["wpe.weight"],6)
def build(T, pk=5.0):
    sd=dict(D); n=0
    for k,v in D.items():
        if v.dim()!=2: continue
        if k=="wpe.weight": sd[k]=wpe_fixed; continue
        z,mu,s=cache[k]; r=v-mu; big=z.abs()>15
        tau=r[big].std().item()
        m=(z.abs()>T)&(r.abs()<pk*tau)
        o=v.clone(); o[m]=mu[m]; sd[k]=o; n+=m.sum().item()
    return sd,n
for T in [8,6,5,4]:
    sd,n=build(T)
    print(f"T={T} flagged={n} ppl={perplexity(sd,ids,max_tokens=8193):.3f}",flush=True)
print(generate(sd,tok,"The capital of France is",15))
