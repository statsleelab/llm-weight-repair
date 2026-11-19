from common import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:1024]
T=8
fx={}
for k in KEYS(sd):
    o,m,med=flag(sd[k],T); fx[k]=torch.where(o,med.expand_as(sd[k]),sd[k])
r=dict(sd)
def ev(tag):
    print(tag,nll_windows(r,ids,window=256),flush=True)
ev("damaged")
for k in ["wte.weight","wpe.weight"]:
    r[k]=fx[k]; ev("fix "+k)
for g in ["c_attn","attn.c_proj","c_fc","mlp.c_proj"]:
    for k in KEYS(sd):
        if g in k and "wte" not in k: r[k]=fx[k]
    ev("+ all "+g)
