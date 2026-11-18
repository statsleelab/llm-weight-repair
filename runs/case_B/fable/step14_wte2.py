import torch, numpy as np, math, time, sys
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights, Tokenizer, perplexity
from repair_lib import posterior
sd = load_weights("damaged.safetensors"); a=load_weights("cand_eb4.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
W=sd["wte.weight"]; m0=W.median(dim=0).values[None,:]
for k in [int(x) for x in sys.argv[1:]]:
    cur=a["wte.weight"]
    for it in range(2):
        torch.manual_seed(0)
        U,s,V=torch.svd_lowrank(cur-m0,q=k+10,niter=3)
        m=m0+(U[:,:k]*s[:k])@V[:,:k].T
        cur,p=posterior(W,m,key="wte.weight",line_z=False)
        M=(p>.5).float(); cc=M.sum(0); f=M.mean().item()
        top=torch.topk(cc,4)
        print(f"k={k} it={it} Ep={p.mean():.5f} f={f:.5f} top cols {[(i.item(),int(cc[i])) for i in top.indices]} exp {f*W.shape[0]:.0f}; resid rstd {((W-m).abs().median()*1.4826).item():.4f} vs {((W-m0).abs().median()*1.4826).item():.4f}",flush=True)
    b=dict(a); b["wte.weight"]=cur
    print("  ppl 2k:",perplexity(b,ids,max_tokens=2048),flush=True)
    torch.save(cur,f"wte_k{k}.pt")
