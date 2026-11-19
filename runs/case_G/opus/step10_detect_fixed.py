import torch; torch.set_num_threads(1)
import numpy as np, json, sys
from gpt2_lib import *
from detect_lib import llr_map
sd=load_weights("damaged.safetensors")
def box(L,h,w):
    c=np.zeros((L.shape[0]+1,L.shape[1]+1)); c[1:,1:]=L.cumsum(0).cumsum(1)
    return c[h:,w:]-c[:-h,w:]-c[h:,:-w]+c[:-h,:-w]
out={}
keys=[k for k,v in sd.items() if v.dim()==2]
for k in keys:
    w=sd[k].numpy(); R,C=w.shape
    L=llr_map(w)
    best=None
    for dh in (0,):
        h=int(round(R*np.sqrt(0.03))); ww=int(round(C*np.sqrt(0.03)))
        S=box(L,h,ww)
        i,j=np.unravel_index(int(S.argmax()),S.shape)
        sc=S[i,j]
        S2=S.copy(); S2[max(0,i-h):i+h, max(0,j-ww):j+ww]=-1e18
        sc2=S2.max()
    out[k]=[int(i),int(i+h),int(j),int(j+ww)]
    flag="  <-- weak" if sc2>0.5*sc else ""
    print("%-26s rows %5d-%5d cols %4d-%4d  score %8.0f  2nd %8.0f%s"%(k,i,i+h,j,j+ww,sc,sc2,flag),flush=True)
json.dump(out,open("blocks.json","w"),indent=1)
