from common import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
zs={k:zscore(sd[k])[0] for k in KEYS(sd)}
for T in [5,7,10,14]:
    r={k:v.clone() for k,v in sd.items()}
    n=0
    for k in KEYS(sd):
        m=zs[k].abs()>T; n+=m.sum().item()
        r[k][m]=0.0
    print("T",T,"flagged",n,"ppl",perplexity(r,ids,max_tokens=4096),flush=True)
