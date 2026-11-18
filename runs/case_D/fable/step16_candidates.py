import torch, math, json
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt",encoding="utf-8").read())[6144:6144+4096]
dam=load_weights("damaged.safetensors"); sd=load_weights("cand_impute.safetensors")
cands={"h.10.attn.c_proj.weight":[(419,238),(442,238),(430,238)],
 "h.9.attn.c_attn.weight":[(145,1606),(144,1648),(144,1686),(144,1727)],
 "h.10.attn.c_attn.weight":[(622,1851),(622,1840),(622,1983)],
 "h.11.attn.c_proj.weight":[(375,514),(0,62)],
 "h.8.attn.c_attn.weight":[(42,1725),(43,1732)],
 "h.4.attn.c_proj.weight":[(364,388),(384,388),(374,388)],
 "h.2.attn.c_attn.weight":[(458,65),(454,29)]}
res={}
for k,cs in cands.items():
    keep=sd[k]; n,m=keep.shape; h,w=round(.1*n),round(.1*m)
    sd[k]=dam[k]; nll,_=nll_windows(sd,ids,window=512); print(f"{k} no-fix: nll {nll:.4f}",flush=True)
    for (i,j) in cs:
        W=dam[k].clone(); W[i:i+h,j:j+w]=0; sd[k]=W
        nll,_=nll_windows(sd,ids,window=512); res[f"{k}@{i},{j}"]=nll
        print(f"{k} zero@({i},{j}): nll {nll:.4f}",flush=True)
    sd[k]=keep
json.dump(res,open("cand_results.json","w"),indent=1)
