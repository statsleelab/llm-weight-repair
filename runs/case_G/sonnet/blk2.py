import torch,numpy as np
def kadane(t):
    best=-1e9;cur=0;s=0;bs=be=0
    for i,v in enumerate(t):
        if cur<=0: cur=v;s=i
        else: cur+=v
        if cur>best: best=cur;bs=s;be=i+1
    return bs,be,best
def slopes(w,rr,cc,axis):
    lw=torch.log(w.abs()+1e-2)
    rm=lw.median(1,keepdim=True).values; cm=lw.median(0,keepdim=True).values
    pred=rm+cm-lw.median()
    x=lw[rr[0]:rr[1],cc[0]:cc[1]]; y=pred[rr[0]:rr[1],cc[0]:cc[1]]
    d=1 if axis==0 else 0   # axis 0: per-row scores, reduce over columns
    x=x-x.mean(d,keepdim=True); y=y-y.mean(d,keepdim=True)
    return ((x*y).sum(d)/((y*y).sum(d)+1e-9))
def find_box(w,coarse_r,coarse_c,iters=3):
    R,C=w.shape
    rr=list(coarse_r); cc=list(coarse_c)
    for _ in range(iters):
        # rows scored over current cols (full row range)
        s=slopes(w,(0,R),(cc[0]+4,cc[1]-4),0)
        mu=s.median().item() if False else s[:].clone().sort().values[int(0.7*R)].item()
        t=(mu/2-s).tolist()
        a,b,_=kadane(t); rr=[a,b]
        s=slopes(w,(rr[0]+2,rr[1]-2),(0,C),1)
        mu=s.sort().values[int(0.7*C)].item()
        t=(mu/2-s).tolist()
        a,b,_=kadane(t); cc=[a,b]
    return rr,cc
