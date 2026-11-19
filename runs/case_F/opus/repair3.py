import torch, math
from repair_lib import wpe_resid, mad
from repair2 import rank1_scale, posterior
def median_polish(v, iters=2):
    row=torch.zeros(v.shape[0],1); col=torch.zeros(1,v.shape[1])
    for _ in range(iters):
        col=col+(v-row-col).median(0,keepdim=True).values
        row=row+(v-row-col).median(1,keepdim=True).values
    return row+col
def repair(sd, mode="hard", thr=0.5, nu=4.0, line_adapt=True, scale="rank1", T=None):
    out={}
    for k,v in sd.items():
        if v.dim()!=2: out[k]=v.clone(); continue
        gm=((v-v.median()).abs().median()*1.4826).item()
        sigc=10*gm
        if k=="wpe.weight":
            sigc=0.1955
            r,pred=wpe_resid(v); s=mad(r,0).clamp_min(1e-4).expand_as(r)
            pi=posterior(r,s,sigc,nu=nu,line_adapt=False)
            v2=v.clone(); m=pi>0.5; v2[m]=pred[m]
            r,pred=wpe_resid(v2); r=v-pred
            pi=posterior(r,s,sigc,nu=nu,line_adapt=False)
            s0=rank1_scale(v[:4].abs()); pi[:4]=posterior(v[:4],s0,sigc,nu=nu,line_adapt=False); pred[:4]=0
        else:
            pred=median_polish(v)
            r=v-pred
            if scale=="rank1": s=rank1_scale(r.abs())
            else: s=torch.maximum(torch.maximum(mad(r,1),mad(r,0)),torch.tensor(gm))
            if T is not None:
                pi=((r/s).abs()>T).float()
            else:
                pi=posterior(r,s,sigc,nu=nu,line_adapt=line_adapt)
        if mode=="hard":
            m=pi>thr; w=v.clone(); w[m]=pred[m]
        else:
            w=(1-pi)*v+pi*pred
        out[k]=w
    return out
