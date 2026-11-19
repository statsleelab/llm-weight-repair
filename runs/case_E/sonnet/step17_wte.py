from common import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:1024]
def fixm(k,T):
    z,_,med=zscore(sd[k]); o=z.abs()>T
    return torch.where(o,med.expand_as(sd[k]),sd[k]),o.sum().item()
base=dict(sd)
for k in KEYS(sd):
    if k!="wte.weight": base[k],_=fixm(k,10)
for T in [3.5,4.5,6,8,12]:
    r=dict(base); r["wte.weight"],n=fixm("wte.weight",T)
    print("wte T",T,"n",n,"nll",nll_windows(r,ids,window=256),flush=True)
