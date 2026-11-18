import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
wp,mw=repair_wpe2(D["wpe.weight"]); print("wpe flagged", mw.sum().item(), "edge rows", mw[:8].sum(1).tolist(), mw[-2:].sum(1).tolist())
for c in [3.0,2.6,2.2]:
    sd=dict(D); n=0; fr=[]
    for k,v in D.items():
        if v.dim()!=2: continue
        if k=="wpe.weight": sd[k]=wp; continue
        sd[k],m,tau=repair3(v,c); n+=m.sum().item(); fr.append((k,round(m.float().mean().item()*100,3)))
    print(f"c={c} flagged={n} ppl={perplexity(sd,ids,max_tokens=4097):.3f}",flush=True)
    print("  ", sorted(fr,key=lambda t:-t[1])[:6])
