import torch,numpy as np,json,sys
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
def ppl(s): 
    r=nll_windows(s,ids,window=512,max_tokens=4096)
    return r
print("damaged",ppl(sd),flush=True)
bx=json.load(open("boxes2.json"))
w=sd["wte.weight"]
# check wte column offset by row/col box corr
lw=torch.log(w.abs()+1e-2)
pred=lw.median(1,keepdim=True).values+lw.median(0,keepdim=True).values-lw.median()
def bc(a,b,bh,bw):
    x=lw[a:a+bh,b:b+bw].flatten();y=pred[a:a+bh,b:b+bw].flatten();x=x-x.mean();y=y-y.mean();return ((x*y).sum()/(x.norm()*y.norm())).item()
print([(b,round(bc(34395,b,8705,133),3)) for b in range(585,650,5)])
