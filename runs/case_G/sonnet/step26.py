import torch,numpy as np,json
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
co=json.load(open("coarse2.json"))
def boxcorr(lw,pred,a,b,bh,bw):
    x=lw[a:a+bh,b:b+bw].flatten(); y=pred[a:a+bh,b:b+bw].flatten()
    x=x-x.mean();y=y-y.mean()
    return ((x*y).sum()/(x.norm()*y.norm()+1e-9)).item()
boxes={}
items=[(k,v) for k,v in sd.items() if v.ndim==2 and k!="wpe.weight"]
for k,w in items:
    R,C=w.shape
    bh=round(0.17320508*R); bw=round(0.17320508*C)
    if k=="wte.weight": a,b=34395,600
    else: a,b=co[k]
    lw=torch.log(w.abs()+1e-2)
    pred=lw.median(1,keepdim=True).values+lw.median(0,keepdim=True).values-lw.median()
    for sweep in range(2):
        span=range(max(0,a-24),min(R-bh,a+24)+1)
        if k=="wte.weight": span=range(max(0,a-40),min(R-bh,a+40)+1)
        a=min(span,key=lambda x:boxcorr(lw,pred,x,b,bh,bw))
        span=range(max(0,b-24),min(C-bw,b+24)+1)
        if k=="wte.weight": span=range(max(0,b-60),min(C-bw,b+60)+1)
        b=min(span,key=lambda x:boxcorr(lw,pred,x,a,bh,bw) if False else boxcorr(lw,pred,a,x,bh,bw))
    boxes[k]=(a,bh,b,bw,boxcorr(lw,pred,a,b,bh,bw))
    print(k,boxes[k],flush=True)
json.dump(boxes,open("boxes2.json","w"))
