from common import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:1024]
for T in [6,10,16]:
    r=dict(sd); n=0
    for k in KEYS(sd):
        z,_,med=zscore(sd[k]); o=z.abs()>T; n+=o.sum().item()
        r[k]=torch.where(o,med.expand_as(sd[k]),sd[k])
    print("T",T,"flagged",n,"nll",nll_windows(r,ids,window=256),flush=True)
