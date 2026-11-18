import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def ppl(sd): return perplexity(sd,ids,max_tokens=2049)
cache={}
for k,v in D.items():
    if v.dim()!=2: continue
    if k=="wte.weight":  # quantile on huge tensor: do in chunks via detect2 only + column calibration
        z,mu,s=detect2(v); r=v-mu
        za=(r/s).abs(); tc=za.kthvalue(int(.98*za.shape[0]),dim=0,keepdim=True).values
        sc=s*(tc/tc.median()).clamp_min(1.0); cache[k]=(r/sc,mu,sc)
    else: cache[k]=detect4(v)
    z,mu,sc=cache[k]; tau,n=estimate_tau(v-mu,sc)
    print(f"{k:26s} tau={tau:.3f} (n={n}) flag5={(z.abs()>5).float().mean()*100:.3f}%",flush=True)
torch.save(cache,"/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/cache4.pt")
