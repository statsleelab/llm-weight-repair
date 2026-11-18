import torch, numpy as np, math, time
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights, Tokenizer, perplexity
from repair_lib import posterior
sd = load_weights("damaged.safetensors"); out=dict(sd)
def prior_mean(k,W):
    if k=="wpe.weight":
        pad=torch.cat([W[:1].expand(4,-1),W,W[-1:].expand(4,-1)],0)
        return pad.unfold(0,9,1).median(dim=-1).values
    if k=="wte.weight": return W.median(dim=0).values[None,:]
    return None
if __name__=="__main__":
    for k,W in sd.items():
        if W.ndim!=2: continue
        dims=(0,) if k in("wte.weight","wpe.weight") else (0,1)
        out[k],p=posterior(W,prior_mean(k,W),key=k,line_dims=dims,ksig=5.0,add=4.0)
    save_weights(out,"cand_eb6.safetensors")
    a=load_weights("cand_eb4.safetensors"); b=out
    tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
    print("eb6 2k:",perplexity(b,ids,max_tokens=2048))
    for g in ["wte","wpe","attn.c_attn","attn.c_proj","mlp.c_fc","mlp.c_proj"]:
        m={k:(b[k] if g in k else a[k]) for k in a}
        print("eb4 with",g,"from eb6:",perplexity(m,ids,max_tokens=2048),flush=True)
