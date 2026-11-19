# Dedicated wpe repair: position embeddings are smooth along the position axis, so a running median over
# neighbouring positions predicts each entry far better than a global baseline.  Flag entries far from the
# running median (robust per-column scale) and replace them by it.  First 8 positions (genuinely non-smooth)
# keep the generic repair.  Also assemble final candidates.
import torch, sys, math
torch.set_num_threads(1)
from gpt2_lib import *
dm=load_weights("damaged.safetensors"); c7=load_weights("c7.safetensors"); c8=load_weights("c8.safetensors")
X=dm["wpe.weight"]; W=9; h=W//2
pad=torch.cat([X[:h].flip(0),X,X[-h:].flip(0)],0)
B=pad.unfold(0,W,1).median(-1).values            # (1024,768) running median over positions
R=X-B
s=(R.abs().median(0,keepdim=True).values/0.6745).clamp_min(1e-4)
for K in (4.0,6.0):
    flag=(R.abs()>K*s); flag[:8]=False
    Y=torch.where(flag,B,X); Y[:8]=c7["wpe.weight"][:8]
    print("K",K,"flagged",flag.float().mean().item(),"diff vs c7 rms",(Y-c7["wpe.weight"]).pow(2).mean().sqrt().item())
    for name,base in (("c7",c7),("c7a8",{k:(c8[k] if "attn" in k else v) for k,v in c7.items()})):
        sd=dict(base); sd["wpe.weight"]=Y
        save_weights(sd,f"d_{name}_wpe{int(K)}.safetensors")
save_weights({k:(c8[k] if "attn" in k else v) for k,v in c7.items()},"d_c7a8.safetensors")
