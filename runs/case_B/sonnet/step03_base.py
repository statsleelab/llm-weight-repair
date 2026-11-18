import torch,numpy as np,sys
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
print(len(ids))
sd=load_weights("damaged.safetensors")
print("damaged",perplexity(sd,ids,max_tokens=4096))
for thr in [0.8,1.0,1.5]:
    s2=dict(sd)
    for k,v in sd.items():
        if v.dim()==2 and k!='wte.weight' or k=='wte.weight':
            s2[k]=torch.where(v.abs()>thr,torch.zeros_like(v),v)
    print("zero>",thr,perplexity(s2,ids,max_tokens=4096))
