import torch,numpy as np,sys
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
sd=load_weights("damaged.safetensors")
def mk(thr,skip=()):
    s2=dict(sd)
    for k,v in sd.items():
        if v.dim()==2 and k not in skip:
            s2[k]=torch.where(v.abs()>thr,torch.zeros_like(v),v)
    return s2
for thr in [0.6,0.4,0.3]:
    print("all",thr,perplexity(mk(thr),ids,max_tokens=1024),flush=True)
print("skip wpe",0.6,perplexity(mk(0.6,('wpe.weight',)),ids,max_tokens=1024))
