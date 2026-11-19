import torch, math
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
def brentq(f,lo,hi):
    for _ in range(200):
        m=math.sqrt(lo*hi)
        if (f(lo)<0)==(f(m)<0): lo=m
        else: hi=m
    return m
for k,v in sd.items():
    if v.dim()!=2: continue
    gm=((v-v.median()).abs().median()*1.4826).item()
    a=v.abs().flatten()
    # fit sigma, p from counts beyond t1 and t2 (assume clean negligible beyond t1)
    t1=8*gm; t2=16*gm
    n1=(a>t1).float().mean().item(); n2=(a>t2).float().mean().item()
    r=n2/n1
    f=lambda s: math.erfc(t2/s/math.sqrt(2))/math.erfc(t1/s/math.sqrt(2))-r
    try:
        s=brentq(f,gm*0.5,gm*1000); p=n1/math.erfc(t1/s/math.sqrt(2))
    except Exception as e: s=float('nan');p=float('nan')
    print(f"{k:26s} gm={gm:.4f} std={v.std():.4f} sigc={s:.4f} sigc/gm={s/gm:.2f} p={p:.4f} max={a.max():.2f} max/sigc={a.max()/s:.1f}")
