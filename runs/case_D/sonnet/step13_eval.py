import torch, numpy as np, sys
torch.set_num_threads(1)
from gpt2_lib import *
from step09_wpe import repair_wpe
from step10_bayes import shrink
sd=load_weights("damaged.safetensors"); tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
N=8192
print("base",perplexity(sd,ids,max_tokens=N),flush=True)
for thr in (4.0,6.0,10.0):
    s2=dict(sd); s2["wpe.weight"]=repair_wpe(sd["wpe.weight"],thr,3)
    print("wpe thr",thr,perplexity(s2,ids,max_tokens=N),flush=True)
s2=dict(sd); s2["wte.weight"]=shrink(sd["wte.weight"],1.0)[0]
print("wte bayes only",perplexity(s2,ids,max_tokens=N),flush=True)
