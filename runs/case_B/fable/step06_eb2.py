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
        pad=torch.cat([W[:1].expand(4,-1),W,W[-1:].expand(4,-1)],0)
        m=pad.unfold(0,9,1).median(dim=-1).values
    elif k=="wte.weight":
        m=W.median(dim=0).values[None,:]
    else:
        m=None
    print(k, "t=%.0f"%(time.time()-t), flush=True)
    What,p=posterior(W,m,verbose=True)
    out[k]=What
save_weights(out,"cand_eb2.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
print("ppl eb2 (8k):", perplexity(out, ids, max_tokens=8192))
