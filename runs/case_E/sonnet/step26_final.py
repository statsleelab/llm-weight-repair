from common import *
import json,sys
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
Ts=json.load(open("best_T.json"))
G={"wte":lambda k:k=="wte.weight","wpe":lambda k:k=="wpe.weight","attn":lambda k:"c_attn" in k,"aproj":lambda k:"attn.c_proj" in k,"fc":lambda k:"c_fc" in k,"mproj":lambda k:"mlp.c_proj" in k}
KC={"wte":None,"wpe":5,"attn":None,"aproj":5,"fc":None,"mproj":5}
r=dict(sd); tot=0
for g,f in G.items():
    for k in KEYS(sd):
        if f(k):
            if KC[g] is None:
                z,_,med=zscore(sd[k]); o=z.abs()>Ts[g[:]]
            else:
                o,_,med=flag(sd[k],Ts[g],KC[g])
            tot+=o.sum().item(); r[k]=torch.where(o,med.expand_as(sd[k]),sd[k])
print("replaced",tot)
save_weights(r,"repaired.safetensors")
n=8192
print("damaged nll",nll_windows(sd,ids,window=512,max_tokens=n)[0])
nl,acc=nll_windows(r,ids,window=512,max_tokens=n); print("repaired nll",nl,"ppl",math.exp(nl),"acc",acc)
print(generate(r,tok,"The capital of France is",15))
