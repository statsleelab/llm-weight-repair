import torch; torch.set_num_threads(1)
import numpy as np, json, sys
from gpt2_lib import *
from impute_lib import impute
sd=load_weights("damaged.safetensors")
blocks=json.load(open("blocks_comb.json"))
rng=np.random.default_rng(0)
for k in sys.argv[1:]:
    w=sd[k].numpy(); R,C=w.shape; a,b,l,e=blocks[k]; h,ww=b-a,e-l
    # choose fake block not overlapping real one
    while True:
        i=rng.integers(0,R-h+1); j=rng.integers(0,C-ww+1)
        if i+h<=a or i>=b or j+ww<=l or j>=e: break
    true=w[i:i+h,j:j+ww].copy()
    res=[]
    for K in [0,8,32,96]:
        f=impute(w,i,i+h,j,j+ww,K)[i:i+h,j:j+ww]
        res.append("k=%d relerr %.3f"%(K,((f-true)**2).sum()/(true**2).sum()))
    print(k,(i,j)," | ".join(res),flush=True)
