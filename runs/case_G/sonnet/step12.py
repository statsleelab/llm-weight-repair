import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
def tiles(w,th,tw):
    R,C=w.shape;R2,C2=R//th*th,C//tw*tw
    t=w[:R2,:C2].reshape(R2//th,th,C2//tw,tw)
    return t
res=[]
for k,w in sd.items():
    if w.ndim!=2 or k=="wpe.weight": continue
    t=tiles(w,16,16)
    s=t.abs().mean((1,3))   # mean abs per tile
    # remove row/col effects: robust z of tile after removing row & col medians
    z=s-s.median(1,keepdim=True).values-s.median(0,keepdim=True).values+s.median()
    sc=(z-z.median()).abs().median()*1.4826
    zz=((z-z.median())/sc)
    big=(zz.abs()>6)
    print(f"{k:28s} tiles {tuple(s.shape)} frac|z|>6: {big.float().mean():.4f} max {zz.abs().max():.1f}")
