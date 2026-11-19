# where do flagged entries cluster (=> genuine heavy structure rather than noise)?
import torch, math
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors"); rp=load_weights("c_mp_g_hard.safetensors")
for k,v in sd.items():
    if v.ndim!=2: continue
    m=(v!=rp[k]); q=m.float().mean().item()
    out=[]
    for ax,name in ((1,"row"),(0,"col")):
        f=m.float().mean(ax); n=m.shape[ax]
        sdv=math.sqrt(q*(1-q)/n)
        z=(f-q)/sdv
        bad=(z>6).nonzero().flatten()
        out.append(f"{name}: n>6sd={len(bad)} maxfrac={f.max().item():.3f} idx={bad[:8].tolist()}")
    big=(v.abs()>4.5*1.0*v.abs().flatten()[::37].double().quantile(0.995).item()/1.383)
    print(f"{k:24s} q={q:.4f} | {out[0]} | {out[1]} | n(|x|>4.5sn)={big.sum().item()} of which flagged {(big&m).sum().item()}")
