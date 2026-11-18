import torch,numpy as np,sys
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
sd=load_weights("damaged.safetensors")

def fix_wpe(w,floor=0.03,k=8):
    w=w.numpy().astype(np.float64).copy()
    T,C=w.shape
    flag=np.zeros_like(w,bool)
    for it in range(3):
        cur=w.copy()
        # interpolate flagged along position
        for c in range(C):
            f=flag[:,c]
            if f.any():
                good=np.where(~f)[0]
                cur[f,c]=np.interp(np.where(f)[0],good,w[good,c])
        # residual vs neighbor mean (excluding self) using cur (flagged replaced)
        nb=np.zeros_like(cur)
        nb[1:-1]=(cur[:-2]+cur[2:])/2; nb[0]=cur[1]; nb[-1]=cur[-2]
        res=np.abs(w-nb)
        mad=np.median(res[2:-2],0)
        thr=np.maximum(floor,k*mad)
        flag=res>thr[None,:]
        flag[0]=False  # position 0 is special, handled separately
    out=w.copy()
    for c in range(C):
        f=flag[:,c]
        if f.any():
            good=np.where(~f)[0]
            out[f,c]=np.interp(np.where(f)[0],good,w[good,c])
    print("wpe flagged",flag.sum(),flag.mean())
    return torch.tensor(out,dtype=torch.float32),flag

def flag_mat(w,k,tabs=0.0):
    a=w.abs()
    rm=a.median(1).values; cm=a.median(0).values; g=a.median()
    s=rm[:,None]*cm[None,:]/g
    return a>torch.maximum(k*s,torch.tensor(tabs))

if __name__=="__main__":
    s2=dict(sd)
    s2['wpe.weight'],_=fix_wpe(sd['wpe.weight'])
    print("wpe only",perplexity(s2,ids,max_tokens=2048),flush=True)
    for k in [5,7,10]:
        s3=dict(s2); n=0
        for key,v in sd.items():
            if v.dim()==2 and key!='wpe.weight':
                f=flag_mat(v,k); n+=f.sum().item()
                s3[key]=torch.where(f,torch.zeros_like(v),v)
        print("k",k,"flagged frac %.4f"%(n/sum(v.numel() for kk,v in sd.items() if v.dim()==2 and kk!='wpe.weight')),"ppl",perplexity(s3,ids,max_tokens=2048),flush=True)
