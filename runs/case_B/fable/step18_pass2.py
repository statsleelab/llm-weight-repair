import torch, numpy as np, math, time
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights, Tokenizer, perplexity
from repair_lib import posterior
sd=load_weights("damaged.safetensors"); a=load_weights("cand_eb4.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def ppl(m): return perplexity(m,ids,max_tokens=2048)

def wpe_repair(W, cur, h=3, iters=3, verbose=True):
    """wpe is smooth along position: prior mean = leave-one-out weighted local average over +-h
    positions of the current estimate, weights = 1-p (so suspected entries do not vote)."""
    w=(W-cur).abs()<1e-6                     # start: entries untouched by pass 1 are trusted
    w=w.float().clamp_min(0.02)
    for it in range(iters):
        num=torch.zeros_like(W); den=torch.zeros_like(W)
        for d in range(1,h+1):
            num[d:]+= (w*W)[:-d]; den[d:]+=w[:-d]
            num[:-d]+=(w*W)[d:];  den[:-d]+=w[d:]
        m=num/den.clamp_min(1e-6)
        m[0]=cur[0]; m[1]=0.5*(m[1]+cur[1])  # position 0 (and 1) are not on the smooth curve
        cur,p=posterior(W,m,key="wpe.weight",line_z=False)
        w=(1-p).clamp_min(1e-3)
        if verbose: print(f"  wpe it{it}: resid rstd {((W-m).abs().median()*1.4826).item():.5f}  Ep={p.mean():.5f} f={(p>.5).float().mean():.5f}")
    return cur

def lowrank_pass(k_, W, cur, key, rank=32):
    torch.manual_seed(0)
    U,s,V=torch.svd_lowrank(cur,q=rank+10,niter=3)
    m=(U[:,:rank]*s[:rank])@V[:,:rank].T
    out,p=posterior(W,m,key=key,line_z=False)
    return out

if __name__=="__main__":
    print("eb4:",ppl(a))
    b=dict(a); b["wpe.weight"]=wpe_repair(sd["wpe.weight"],a["wpe.weight"])
    print("eb4 + wpe smooth:",ppl(b),flush=True)
    b["wte.weight"]=torch.load("wte_k32.pt")
    print(" + wte k32:",ppl(b),flush=True)
    t=time.time()
    c=dict(b)
    for k,W in sd.items():
        if W.ndim!=2 or k in("wte.weight","wpe.weight"): continue
        c[k]=lowrank_pass(k,W,a[k],k,32)
    print(" + lowrank32 prior for all blocks:",ppl(c),time.time()-t,flush=True)
    for grp in ["attn.c_attn","attn.c_proj","mlp.c_fc","mlp.c_proj"]:
        m={k:(c[k] if grp in k else b[k]) for k in b}
        print("   only",grp,":",ppl(m),flush=True)
    save_weights(b,"cand_p2b.safetensors"); save_weights(c,"cand_p2c.safetensors")
