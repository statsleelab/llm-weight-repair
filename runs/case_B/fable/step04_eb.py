import torch, numpy as np, math, time
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights, Tokenizer, perplexity
from repair_lib import posterior
sd = load_weights("damaged.safetensors")
out = dict(sd)
t=time.time()
for k,W in sd.items():
    if W.ndim!=2: continue
    if k=="wpe.weight":
        # smooth in position: local median over +-4 positions as prior mean
        pad=torch.cat([W[:1].expand(4,-1),W,W[-1:].expand(4,-1)],0)
        m=pad.unfold(0,9,1).median(dim=-1).values
    elif k=="wte.weight":
        m=W.median(dim=0).values[None,:]
    else:
        m=None
    What,p=posterior(W,m,sig_n=10*W.std().item()/math.sqrt(2),verbose=(k in("wte.weight","wpe.weight","h.0.attn.c_proj.weight","h.6.mlp.c_fc.weight")))
    out[k]=What
    print(f"{k:26s} E[#corrupt]/N={p.mean():.5f} p>.5: {(p>.5).float().mean():.5f}  t={time.time()-t:.0f}s",flush=True)
save_weights(out,"cand_eb1.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
print("ppl eb1 (8k):", perplexity(out, ids, max_tokens=8192))
