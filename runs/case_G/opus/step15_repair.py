import torch; torch.set_num_threads(1)
import numpy as np, json, sys
from gpt2_lib import *
from impute_lib import impute
blocks=json.load(open(sys.argv[1])); out=sys.argv[2]
sd=load_weights("damaged.safetensors")
blocks["wpe.weight"]=[125,302,591,724]
for k,(a,b,l,e) in blocks.items():
    w=sd[k].numpy()
    if k=="wpe.weight": K=8
    elif k.endswith("attn.c_proj.weight"): K=0
    elif k=="wte.weight": K=32
    else: K=32
    f=impute(w,a,b,l,e,K,iters=8 if k=="wte.weight" else 15)
    sd[k]=torch.tensor(f)
    print(k,(a,b,l,e),"K",K,flush=True)
save_weights(sd,out)
