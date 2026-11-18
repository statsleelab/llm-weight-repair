import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
def repair_wpe(w, thr=6.0, iters=4):
    w=w.clone(); T=w.shape[0]
    good=torch.ones_like(w,dtype=torch.bool)
    for it in range(iters):
        pred=torch.zeros_like(w)
        # local estimate: average of up to 2 nearest good neighbours each side using linear interpolation via mean of neighbors
        for i in range(1,T-1):
            ws=[];
            for off in (-2,-1,1,2):
                j=i+off
                if 0<=j<T: ws.append((w[j],good[j]))
            vals=torch.stack([a for a,_ in ws]); g=torch.stack([b for _,b in ws]).float()
            # nearest-neighbour weighted mean (weights 2 for adjacent)
            wt=g*torch.tensor([1.,2.,2.,1.])[:len(ws)].unsqueeze(1) if len(ws)==4 else g
            pred[i]=(vals*wt).sum(0)/wt.sum(0).clamp(min=1e-6)
        r=(w-pred)[1:-1]
        cs=r.abs().median(0).values*1.4826+1e-6
        z=(w-pred)/cs
        flag=z.abs()>thr
        flag[0]=False; flag[1]=False; flag[-1]=False
        if it==iters-1 or True:
            pass
        good=~flag
        print("iter",it,"flagged",flag.sum().item())
        w=torch.where(flag,pred,w)
    return w
if __name__=="__main__":
    sd=load_weights("damaged.safetensors"); tok=Tokenizer("tokenizer")
    ids=tok.encode(open("dev.txt").read())
    for thr in (6.0,):
        new=repair_wpe(sd["wpe.weight"],thr)
        sd2=dict(sd); sd2["wpe.weight"]=new
        print(thr,"ppl",perplexity(sd2,ids,max_tokens=4096))
