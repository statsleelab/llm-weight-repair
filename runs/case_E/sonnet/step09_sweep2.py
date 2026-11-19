from common import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:2048]
for T in [5,7,10]:
    r=dict(sd); n=0;n0=0
    for k in KEYS(sd):
        o,m,med=flag(sd[k],T); n+=o.sum().item(); n0+=m.sum().item()
        r[k]=sd[k].clone(); r[k]=torch.where(o,med.expand_as(r[k]),r[k])
    print("T",T,"flagged",n,"of",n0,"nll",nll_windows(r,ids,window=256),flush=True)
