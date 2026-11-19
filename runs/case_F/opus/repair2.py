import torch, math
from repair_lib import wpe_resid, mad
def rank1_scale(a, iters=4):
    r=torch.ones(a.shape[0],1); c=torch.ones(1,a.shape[1])
    for _ in range(iters):
        c=(a/r).median(0,keepdim=True).values.clamp_min(1e-8)
        r=(a/c).median(1,keepdim=True).values.clamp_min(1e-8)
    return r*c*1.4826
def t_logpdf(z, nu):
    return math.lgamma((nu+1)/2)-math.lgamma(nu/2)-0.5*math.log(nu*math.pi)-(nu+1)/2*torch.log1p(z*z/nu)
def posterior(v, s, sigc, p=0.03, nu=4.0, iters=3, line_adapt=True):
    lc_base=-math.log(sigc)-0.5*math.log(2*math.pi)-0.5*(v/sigc)**2
    l0=t_logpdf(v/s,nu)-torch.log(s)
    P=torch.full_like(v,p)
    for _ in range(iters):
        pi=torch.sigmoid(torch.log(P)+lc_base-torch.log1p(-P)-l0)
        if not line_adapt: break
        Nr=pi.sum(1,keepdim=True); Nc=pi.sum(0,keepdim=True)
        Er=p*v.shape[1]; Ec=p*v.shape[0]
        P=p*(Er/Nr).clamp(max=1)*(Ec/Nc).clamp(max=1)
        P=P.clamp(1e-6,0.5)
    return pi
def repair(sd, mode="hard", thr=0.5, nu=4.0, line_adapt=True, wpe_mode="smooth"):
    out={}
    for k,v in sd.items():
        if v.dim()!=2: out[k]=v.clone(); continue
        gm=((v-v.median()).abs().median()*1.4826).item()
        sigc=10*gm
        if k=="wpe.weight":
            r,pred=wpe_resid(v)
            s=mad(r,0).clamp_min(1e-4)
            sigc=0.1955  # fitted from tail counts (step13)
            pi=posterior(r,s.expand_as(r),sigc,nu=nu,line_adapt=False)
            v2=v.clone(); m=pi>0.5; v2[m]=pred[m]
            r,pred=wpe_resid(v2); r=v-pred
            pi=posterior(r,s.expand_as(r),sigc,nu=nu,line_adapt=False)
            # first rows: neighbour prediction unreliable -> use 0-prior magnitude model
            s0=rank1_scale(v.abs())[:4]
            pi0=posterior(v[:4],s0,sigc,nu=nu,line_adapt=False)
            pi[:4]=pi0; pred[:4]=0
        else:
            s=rank1_scale(v.abs())
            pi=posterior(v,s,sigc,nu=nu,line_adapt=line_adapt)
            pred=torch.zeros_like(v)
        if mode=="hard":
            m=pi>thr; w=v.clone(); w[m]=pred[m]
        else:
            w=(1-pi)*v+pi*pred
        out[k]=w
    return out
