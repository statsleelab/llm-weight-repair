import torch; torch.set_num_threads(1)
from gpt2_lib import *
import repair_lib5 as R
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
sd=load_weights("damaged.safetensors")
r0,_=R.repair(sd,T=8)
W=sd["wte.weight"]; cm=W.median(0,keepdim=True).values; u=W-cm
cs=u.abs().median(0,keepdim=True).values/0.6745
mw=(u/cs).abs()>12
o=dict(sd); o["wpe.weight"]=r0["wpe.weight"]; o["wte.weight"]=torch.where(mw, cm.expand_as(W), W)
n=int(mw.sum())+int((r0["wpe.weight"]!=sd["wpe.weight"]).sum())
for k,v in sd.items():
    if v.dim()!=2 or k in("wpe.weight","wte.weight"): continue
    m,hr,hc,c=R.detect(v,16,protect=False); o[k]=torch.where(m,c.expand_as(v),v); n+=int(m.sum())
print("entries changed", n)
save_weights(o,"repaired.safetensors")
print("repaired ppl(2048)", perplexity(o, ids, max_tokens=2048))
print("damaged ppl(2048)", perplexity(sd, ids, max_tokens=2048))
print(generate(o,tok,"The capital of France is",12))
