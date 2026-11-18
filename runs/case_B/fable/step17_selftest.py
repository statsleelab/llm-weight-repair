"""Self-consistency test: treat candidate eb4 as 'clean', re-apply the diagnosed corruption
(1% entries + N(0,(10 std)^2)), re-run the same repair, see how much ppl is lost and in which group."""
import torch, numpy as np, math, time
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights, Tokenizer, perplexity
from repair_lib import posterior
from step13_eb6 import prior_mean
a=load_weights("cand_eb4.safetensors")
g=torch.Generator().manual_seed(123)
rep={}; stats={}
for k,W in a.items():
    if W.ndim!=2: continue
    mask=torch.rand(W.shape,generator=g)<0.01
    Wc=W+mask*torch.randn(W.shape,generator=g)*10*W.std()
    rep[k],p=posterior(Wc,prior_mean(k,Wc),key=k,line_z=False)
    err=rep[k]-W
    stats[k]=(err[mask].pow(2).sum()/W.pow(2).sum()).item(), (err[~mask].pow(2).sum()/W.pow(2).sum()).item()
for grp in ["wte","wpe","attn.c_attn","attn.c_proj","mlp.c_fc","mlp.c_proj"]:
    ks=[k for k in stats if grp in k]
    print(grp,"rel sq err from corrupted entries %.4f, from clean entries %.4f"%(np.mean([stats[k][0] for k in ks]),np.mean([stats[k][1] for k in ks])))
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
print("base eb4 2k:",perplexity(a,ids,max_tokens=2048))
m=dict(a); m.update(rep)
print("all re-corrupted+repaired:",perplexity(m,ids,max_tokens=2048),flush=True)
for grp in ["wte","wpe","attn.c_attn","attn.c_proj","mlp.c_fc","mlp.c_proj"]:
    m={k:(rep[k] if grp in k else a[k]) for k in a}
    print("only",grp,":",perplexity(m,ids,max_tokens=2048),flush=True)
