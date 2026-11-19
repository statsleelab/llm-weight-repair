import torch; torch.set_num_threads(1)
from gpt2_lib import *
import repair_lib5 as R
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
sd=load_weights("damaged.safetensors")
def run(T, hk, prot):
    out=dict(sd)
    for k,v in sd.items():
        if v.dim()!=2 or k=="wpe.weight": continue
        m,hr,hc,cm=R.detect(v,T,heavy_k=hk,protect=prot and k!="wte.weight")
        out[k]=torch.where(m,cm.expand_as(v),v)
    r,_=R.repair(sd,T=T); out["wpe.weight"]=r["wpe.weight"]
    return out
for T,hk,prot in [(8,4,True),(8,4,False),(8,20,True),(12,4,False),(16,4,False)]:
    print(T,hk,prot, perplexity(run(T,hk,prot), ids, max_tokens=1024), flush=True)
