import torch, math; torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
emb=load_weights("emb_fixed.safetensors")
lay={**load_weights("layers_0_6.safetensors"),**load_weights("layers_6_12.safetensors")}
A=dict(sd); A.update(emb)
B=dict(A); B.update(lay)
C=dict(A)
for k,v in lay.items():
    m=(sd[k]==0); d=v[m]; # recompute capped: scale imputed values down when alpha>1 approx by norm ratio not available; use clip to row-col robust scale
    C[k]=v
import json
log={**json.load(open("layers_0_6.json")),**json.load(open("layers_6_12.json"))}
for k,v in lay.items():
    a=log[k]["alpha"]
    if a>1:
        m=(sd[k]==0); w=v.clone(); w[m]=v[m]/a; C[k]=w
for name,s in [("emb only",A),("emb+layers",B),("emb+layers alpha<=1",C)]:
    nll,acc=nll_windows(s,ids,max_tokens=8192); print(name,"ppl",round(math.exp(nll),3),"acc",round(acc,4),flush=True)
save_weights(B,"cand_B.safetensors"); save_weights(C,"cand_C.safetensors")
