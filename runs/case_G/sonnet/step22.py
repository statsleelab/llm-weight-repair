import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
from blk import *
sd=load_weights("damaged.safetensors")
for k,w in sd.items():
    if w.ndim!=2 or k=="wpe.weight": continue
    th=64 if w.shape[0]>=3072 else 32
    tw=64 if w.shape[1]>=3072 else 32
    c=tilecorr(w,th,tw)
    med=c.median().item()
    low=(c<0.35*med)|(c<0.05)
    ii=low.nonzero()
    if len(ii)==0: print(k,"none med %.2f"%med); continue
    r0,r1=ii[:,0].min().item(),ii[:,0].max().item(); c0,c1=ii[:,1].min().item(),ii[:,1].max().item()
    print(f"{k:26s} med{med:.2f} nlow {len(ii)} rows {r0*th}-{(r1+1)*th} cols {c0*tw}-{(c1+1)*tw} boxfill {len(ii)/((r1-r0+1)*(c1-c0+1)):.2f}")
