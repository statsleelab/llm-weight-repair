from common import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
ids=ids[:2048]
T=8
zs={k:zscore(sd[k])[0] for k in KEYS(sd)}
def fix(keys):
    r=dict(sd)
    for k in keys:
        r[k]=sd[k].clone(); r[k][zs[k].abs()>T]=0
    return r
groups={"emb":["wte.weight","wpe.weight"],
 "attn":[k for k in KEYS(sd) if "c_attn" in k],
 "aproj":[k for k in KEYS(sd) if "attn.c_proj" in k],
 "fc":[k for k in KEYS(sd) if "c_fc" in k],
 "mproj":[k for k in KEYS(sd) if "mlp.c_proj" in k]}
allk=KEYS(sd)
for name,g in groups.items():
    # fix everything except this group
    r=fix([k for k in allk if k not in g])
    print("all fixed except",name,nll_windows(r,ids,window=256),flush=True)
