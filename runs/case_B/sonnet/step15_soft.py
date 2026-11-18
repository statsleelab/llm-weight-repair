import torch,numpy as np,math,sys
torch.set_num_threads(1)
from gpt2_lib import *
from step12_repair import fix_wpe
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
sd=load_weights("damaged.safetensors")
PI=0.0105; SC=1.2
def gg(x,a,b):  # generalized gaussian pdf with per-entry scale a
    return b/(2*a*math.gamma(1/b))*torch.exp(-(x.abs()/a)**b)
def corr_pdf(x): return torch.exp(-0.5*(x/SC)**2)/(SC*math.sqrt(2*math.pi))
def scales(w):
    a=w.abs(); rm=a.median(1).values; cm=a.median(0).values; g=a.median()
    return rm[:,None]*cm[None,:]/g
def fit(w,s):
    g=torch.Generator().manual_seed(0)
    n=w.numel(); idx=torch.randperm(n,generator=g)[:600000]
    x=w.flatten()[idx]; ss=s.flatten()[idx]
    best=None
    for b in [1.5,1.8,2.0,2.3,2.7]:
        for a0 in np.linspace(1.4,3.4,21):
            a=a0*ss
            ll=torch.log((1-PI)*gg(x,a,b)+PI*corr_pdf(x)).sum().item()
            if best is None or ll>best[0]: best=(ll,b,a0)
    return best[1],best[2]
def posterior_clean(w,s,b,a0):
    a=a0*s
    lc=math.log(1-PI)+math.log(b/(2*math.gamma(1/b)))-torch.log(a)-(w.abs()/a)**b
    lo=math.log(PI)-0.5*(w/SC)**2-math.log(SC*math.sqrt(2*math.pi))
    return torch.sigmoid(lc-lo)
def repair(sd,wpe_fix=True,power=1.0,verbose=True):
    s2=dict(sd)
    if wpe_fix: s2['wpe.weight'],_=fix_wpe(sd['wpe.weight'])
    for key,v in sd.items():
        if v.dim()==2 and key!='wpe.weight':
            s=scales(v); b,a0=fit(v,s)
            p=posterior_clean(v,s,b,a0)
            if verbose: print(key,"beta",b,"a0 %.2f"%a0,"exp corrupted %.4f"%(1-p).mean().item(),flush=True)
            s2[key]=v*p**power
    return s2
if __name__=="__main__":
    s2=repair(sd)
    torch.save({k:v for k,v in s2.items()},"/dev/null") if False else None
    print("ppl",perplexity(s2,ids,max_tokens=2048))
    save_weights(s2,"cand_soft.safetensors")
