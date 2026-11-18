import torch, numpy as np, sys
torch.set_num_threads(1)
from gpt2_lib import *
from step10_bayes import shrink
sd=load_weights("damaged.safetensors"); tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
groups={"wte":["wte.weight"],"c_attn":[f"h.{i}.attn.c_attn.weight" for i in range(12)],
"attn_proj":[f"h.{i}.attn.c_proj.weight" for i in range(12)],"c_fc":[f"h.{i}.mlp.c_fc.weight" for i in range(12)],
"mlp_proj":[f"h.{i}.mlp.c_proj.weight" for i in range(12)]}
for g,ks in groups.items():
    sd2=dict(sd)
    for k in ks: sd2[k]=shrink(sd[k],1.0)[0]
    print(g,perplexity(sd2,ids,max_tokens=2048),flush=True)
print("base",perplexity(sd,ids,max_tokens=2048))
