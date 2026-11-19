import torch; torch.set_num_threads(1)
from gpt2_lib import *
import repair_lib5 as R
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
sd=load_weights("damaged.safetensors")
r0,_=R.repair(sd,T=8)
W=sd["wte.weight"]; cm=W.median(0,keepdim=True).values; u=W-cm
cs=u.abs().median(0,keepdim=True).values/0.6745
wte_fix=torch.where((u/cs).abs()>12, cm.expand_as(W), W)
def build(T):
    o=dict(sd); o["wpe.weight"]=r0["wpe.weight"]; o["wte.weight"]=wte_fix
    for k,v in sd.items():
        if v.dim()!=2 or k in("wpe.weight","wte.weight"): continue
        m,hr,hc,c=R.detect(v,T,protect=False); o[k]=torch.where(m,c.expand_as(v),v)
    return o
for T in [10,24]:
    print(T, perplexity(build(T), ids, max_tokens=1024), flush=True)
