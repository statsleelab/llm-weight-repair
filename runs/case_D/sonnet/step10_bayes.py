import torch, numpy as np, sys
torch.set_num_threads(1)
from gpt2_lib import *
from step09_wpe import repair_wpe

def scale_model(w, iters=3):
    a=w.abs()+1e-8
    r=torch.ones(w.shape[0],1); c=torch.ones(1,w.shape[1])
    for _ in range(iters):
        c=(a/r).median(0,keepdim=True).values
        r=(a/c).median(1,keepdim=True).values
    s=r*c/0.6745
    return s
def shrink(w, tau_mult=1.0, eps=0.01, tauabs=None, mode="bayes"):
    s=scale_model(w)
    tau=tauabs if tauabs is not None else tau_mult*w.std().item()
    x=w
    lc=-0.5*(x/s)**2-torch.log(s)
    t2=s**2+tau**2
    lk=-0.5*x**2/t2-0.5*torch.log(t2)
    pc=torch.sigmoid((lk+np.log(eps))-(lc+np.log(1-eps)))
    out=x*(1-pc)+pc*x*s**2/t2
    return out,pc
if __name__=="__main__":
    sd=load_weights("damaged.safetensors"); tok=Tokenizer("tokenizer")
    ids=tok.encode(open("dev.txt").read())
    keys=[k for k,v in sd.items() if v.dim()==2 and k!="wpe.weight"]
    for tm in [float(a) for a in sys.argv[1:]]:
        sd2=dict(sd); sd2["wpe.weight"]=repair_wpe(sd["wpe.weight"],6.0,3)
        tot=0
        for k in keys:
            new,pc=shrink(sd[k],tm); sd2[k]=new; tot+=pc.sum().item()
        print("tau_mult",tm,"expected corrupt count",tot,"ppl",perplexity(sd2,ids,max_tokens=4096),flush=True)
