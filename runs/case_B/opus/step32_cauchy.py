import torch, math; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def ppl(sd): return perplexity(sd,ids,max_tokens=4097)
cache=torch.load("/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/cache2.pt")
def post(w,z,mu,s,nu,pi,tau):
    r=w-mu
    lc=(math.lgamma((nu+1)/2)-math.lgamma(nu/2)-0.5*math.log(nu*math.pi))-torch.log(s)-(nu+1)/2*torch.log1p((r/s)**2/nu)
    lk=-math.log(math.pi*tau)-torch.log1p((w/tau)**2)          # Cauchy corruption density
    p=torch.sigmoid(math.log(pi)+lk-math.log(1-pi)-lc)
    return p
def massive_protect(k,w):
    m=torch.zeros_like(w,dtype=torch.bool)
    if "mlp.c_proj" in k and int(k.split(".")[1])<=3:
        for c in (138,447,378): m[:,c]|=w[:,c].abs()>4
    return m
def build(nu,pi,tau,prot):
    sd=dict(D)
    for k,(z,mu,s) in cache.items():
        w=D[k]; p=post(w,z,mu,s,nu,pi,tau)
        if prot: p=p.masked_fill(massive_protect(k,w),0.0)
        sd[k]=mu+(1-p)*(w-mu)
    return sd
best=None
for nu,pi,tau,prot in [(4,0.01,1.0,True),(4,0.01,1.0,False),(4,0.03,1.0,True),(6,0.01,1.0,True)]:
    sd=build(nu,pi,tau,prot); p=ppl(sd); print(f"nu={nu} pi={pi} tau={tau} prot={prot} ppl4096={p:.2f}",flush=True)
    if best is None or p<best[0]: best=(p,(nu,pi,tau,prot),sd)
print("best",best[:2])
if best[0]<4524: save_weights(best[2],"repaired.safetensors"); print("saved")
