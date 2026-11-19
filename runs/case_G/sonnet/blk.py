import torch,numpy as np
def tilecorr(w,th,tw):
    lw=torch.log(w.abs()+1e-2)
    R,C=w.shape
    rm=lw.median(1,keepdim=True).values; cm=lw.median(0,keepdim=True).values
    pred=rm+cm-lw.median()
    R2,C2=R//th*th,C//tw*tw
    a=lw[:R2,:C2].reshape(R2//th,th,C2//tw,tw).permute(0,2,1,3).reshape(R2//th,C2//tw,-1)
    b=pred[:R2,:C2].reshape(R2//th,th,C2//tw,tw).permute(0,2,1,3).reshape(R2//th,C2//tw,-1)
    a=a-a.mean(-1,keepdim=True); b=b-b.mean(-1,keepdim=True)
    return (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)+1e-9)
