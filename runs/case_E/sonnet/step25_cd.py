from common import *
import json
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:1536]
G={"wte":lambda k:k=="wte.weight","wpe":lambda k:k=="wpe.weight","attn":lambda k:"c_attn" in k,"aproj":lambda k:"attn.c_proj" in k,"fc":lambda k:"c_fc" in k,"mproj":lambda k:"mlp.c_proj" in k}
cache={}
def fixed(k,T,kc):
    key=(k,T,kc)
    if key not in cache:
        if kc is None:
            z,_,med=zscore(sd[k]); o=z.abs()>T
        else:
            o,_,med=flag(sd[k],T,kc)
        cache[key]=torch.where(o,med.expand_as(sd[k]),sd[k])
    return cache[key]
KC={"wte":None,"wpe":5,"attn":None,"aproj":5,"fc":None,"mproj":5}
def build(Ts):
    r=dict(sd)
    for g,f in G.items():
        for k in KEYS(sd):
            if f(k): r[k]=fixed(k,Ts[g],KC[g])
    return r
Ts={g:14 for g in G}
best=nll_windows(build(Ts),ids,window=256)[0]; print("start",best,flush=True)
for rnd in range(2):
    for g in G:
        for T in [7,10,14,20,30]:
            if T==Ts[g]: continue
            t=dict(Ts); t[g]=T
            v=nll_windows(build(t),ids,window=256)[0]
            if v<best-0.02: best,Ts=v,t; print("improve",g,T,round(v,3),flush=True)
        cache.clear() if len(cache)>150 else None
print(Ts,best)
json.dump(Ts,open("best_T.json","w"))
