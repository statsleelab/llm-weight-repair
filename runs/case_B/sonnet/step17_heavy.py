import torch,numpy as np,sys
torch.set_num_threads(1)
from gpt2_lib import *
from step15_soft import *
def heavy_masks(v,s,a0,zt=5.0,frac=0.03):
    z=v.abs()/(a0*s)
    big=(z>zt).float()
    rowh=big.mean(1)>frac; colh=big.mean(0)>frac
    return rowh,colh
def repair2(sd,frac=0.03,absk=5.0,verbose=False):
    s2=dict(sd); s2['wpe.weight'],_=fix_wpe(sd['wpe.weight'])
    for key,v in sd.items():
        if v.dim()==2 and key!='wpe.weight':
            s=scales(v); b,a0=fit(v,s)
            p=posterior_clean(v,s,b,a0)
            rowh,colh=heavy_masks(v,s,a0,frac=frac)
            keep=rowh[:,None]|colh[None,:]|(v.abs()>absk)
            p=torch.where(keep,torch.ones_like(p),p)
            if verbose: print(key,"heavy rows",rowh.sum().item(),"cols",colh.sum().item(),"exp corr %.4f"%(1-p).mean().item(),flush=True)
            s2[key]=v*p
    return s2
if __name__=="__main__":
    s2=repair2(sd,verbose=True)
    print("ppl",perplexity(s2,ids,max_tokens=2048))
    save_weights(s2,"cand_heavy.safetensors")
