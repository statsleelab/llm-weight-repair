import torch; torch.set_num_threads(1)
from gpt2_lib import *
import repair_lib5 as R
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
sd=load_weights("damaged.safetensors")
r0,_=R.repair(sd,T=8)
def run(T, Twte):
    out=dict(sd); out["wpe.weight"]=r0["wpe.weight"]; n=0
    for k,v in sd.items():
        if v.dim()!=2 or k=="wpe.weight": continue
        m,hr,hc,cm=R.detect(v,Twte if k=="wte.weight" else T,protect=False)
        out[k]=torch.where(m,cm.expand_as(v),v); n+=int(m.sum())
    return out,n
for T,Tw in [(16,24),(16,32),(16,64),(12,32),(24,32)]:
    o,n=run(T,Tw); print(T,Tw,n, perplexity(o, ids, max_tokens=1024), flush=True)
