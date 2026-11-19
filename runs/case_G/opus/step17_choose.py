import torch; torch.set_num_threads(1)
import numpy as np, json, sys
from gpt2_lib import *
from impute_lib import impute
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
dmg=load_weights("damaged.safetensors"); rep=load_weights("repaired.safetensors")
cands=json.loads(open(sys.argv[1]).read())
def ev(sd): return nll_windows(sd,ids,window=512,max_tokens=2049)[0]
for k,lst in cands.items():
    w=dmg[k].numpy(); R,C=w.shape
    h=int(round(R*np.sqrt(0.03))); ww=int(round(C*np.sqrt(0.03)))
    K=0 if k.endswith("attn.c_proj.weight") else 32
    res=[]
    orig=rep[k]
    for (i,j) in lst:
        rep[k]=torch.tensor(impute(w,i,i+h,j,j+ww,K))
        res.append(((i,j),round(ev(rep),4)))
    rep[k]=orig
    print(k,res,flush=True)
