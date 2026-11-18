import torch, numpy as np, math, time
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights, Tokenizer, perplexity
from repair_lib import posterior
sd = load_weights("damaged.safetensors"); out=dict(sd)
for k,W in sd.items():
    if W.ndim!=2: continue
    if k=="wpe.weight":
        pad=torch.cat([W[:1].expand(4,-1),W,W[-1:].expand(4,-1)],0)
        m=pad.unfold(0,9,1).median(dim=-1).values
    elif k=="wte.weight":
        m=W.median(dim=0).values[None,:]
    else: m=None
    What,p=posterior(W,m,key=k); out[k]=What
    M=(p>.5).float()
    f=M.mean().item()
    rc=M.sum(1); cc=M.sum(0)
    er=f*W.shape[1]; ec=f*W.shape[0]
    zr=(rc-er)/math.sqrt(er); zc=(cc-ec)/math.sqrt(ec)
    tr=torch.topk(zr,2); tc=torch.topk(zc,2)
    print(f"{k:24s} Ep={p.mean():.5f} f={f:.5f} rows z>6: {(zr>6).sum().item():4d} top {[(i.item(),int(rc[i])) for i in tr.indices]} exp {er:.1f} | cols z>6: {(zc>6).sum().item():4d} top {[(i.item(),int(cc[i])) for i in tc.indices]} exp {ec:.1f}",flush=True)
save_weights(out,"cand_eb4.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
print("ppl eb4 (8k):", perplexity(out, ids, max_tokens=8192))
