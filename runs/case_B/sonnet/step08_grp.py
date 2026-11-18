import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:1024]
sd=load_weights("damaged.safetensors")
groups=['wte','wpe','c_attn','attn.c_proj','c_fc','mlp.c_proj']
def mk(sel,thr=1.0):
    s2=dict(sd)
    for k,v in sd.items():
        if v.dim()==2 and any(g in k for g in sel):
            s2[k]=torch.where(v.abs()>thr,torch.zeros_like(v),v)
    return s2
for g in groups: print("only",g,"%.3g"%perplexity(mk([g]),ids),flush=True)
print("all but wpe","%.3g"%perplexity(mk(['wte','c_attn','attn.c_proj','c_fc','mlp.c_proj']),ids))
