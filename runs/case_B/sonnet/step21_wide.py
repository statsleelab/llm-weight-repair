import torch,numpy as np,sys,math
torch.set_num_threads(1)
from gpt2_lib import *
from step15_soft import *
SW=2.0
def repair3(sd,zt=5.0,base=0.008,absk=5.5,sw=SW,verbose=False):
    s2=dict(sd); s2['wpe.weight'],_=fix_wpe(sd['wpe.weight'])
    for key,v in sd.items():
        if v.dim()==2 and key!='wpe.weight':
            s=scales(v); b,a0=fit(v,s); a=a0*s
            z=v.abs()/a
            big=(z>zt).float()
            rf=big.mean(1); cf=big.mean(0)
            om=torch.clamp(torch.maximum(rf[:,None],cf[None,:])-base,0,0.6)
            lcl=torch.log((1-om)*torch.exp(-(z**b))*b/(2*a*math.gamma(1/b)) + 1e-300) if False else None
            # compute in log domain
            lg_clean=-(z**b)+math.log(b/(2*math.gamma(1/b)))-torch.log(a)
            lg_wide=-0.5*(v/sw)**2-math.log(sw*math.sqrt(2*math.pi))
            lc=torch.logaddexp(torch.log1p(-om)+lg_clean, torch.log(om+1e-12)+lg_wide)+math.log(1-PI)
            lo=math.log(PI)-0.5*(v/SC)**2-math.log(SC*math.sqrt(2*math.pi))
            p=torch.sigmoid(lc-lo)
            p=torch.where(v.abs()>absk,torch.ones_like(p),p)
            if verbose: print(key,"om>0 frac %.3f"%(om>0).float().mean().item(),"exp corr %.4f"%(1-p).mean().item(),flush=True)
            s2[key]=v*p
    return s2
if __name__=="__main__":
    s2=repair3(sd,verbose=True)
    print("ppl",perplexity(s2,ids,max_tokens=2048))
    save_weights(s2,"cand_wide.safetensors")
