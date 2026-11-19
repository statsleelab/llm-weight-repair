import torch; torch.set_num_threads(1)
from gpt2_lib import *
import repair_lib5 as R
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
sd=load_weights("damaged.safetensors")
r0,_=R.repair(sd,T=8)
base=dict(sd); base["wpe.weight"]=r0["wpe.weight"]
for k,v in sd.items():
    if v.dim()!=2 or k in("wpe.weight","wte.weight"): continue
    m,hr,hc,cm=R.detect(v,16,protect=False); base[k]=torch.where(m,cm.expand_as(v),v)
torch.save({k:base[k] for k in base if k!="wte.weight"}, "cache_base16.pt")
W=sd["wte.weight"]; cm=W.median(0,keepdim=True).values; u=W-cm
cs=u.abs().median(0,keepdim=True).values/0.6745
zc=u/cs
s=W.abs().median()/0.6745
for T in [8,12,16,24]:
    for gmin in [0,2]:
        m=(zc.abs()>T)&((u.abs()/s)>gmin)
        o=dict(base); o["wte.weight"]=torch.where(m,cm.expand_as(W),W)
        print(T,gmin,int(m.sum()), perplexity(o, ids, max_tokens=1024), flush=True)
