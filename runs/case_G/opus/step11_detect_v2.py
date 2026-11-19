import torch; torch.set_num_threads(1)
import numpy as np, json, sys
from gpt2_lib import *
from detect_lib import llr_map
sd=load_weights("damaged.safetensors")
def box(L,h,w):
    c=np.zeros((L.shape[0]+1,L.shape[1]+1)); c[1:,1:]=L.cumsum(0).cumsum(1)
    return c[h:,w:]-c[:-h,w:]-c[h:,:-w]+c[:-h,:-w]
def lowrank_score(w,k=32):
    w=w.astype(np.float64)
    U,S,Vt=np.linalg.svd(w,full_matrices=False)
    res=w-(U[:,:k]*S[:k])@Vt[:k]
    return res
keys=sys.argv[1:] if len(sys.argv)>1 else [k for k,v in sd.items() if v.dim()==2 and k!="wte.weight"]
old=json.load(open("blocks.json")); out=dict(old)
for k in keys:
    w=sd[k].numpy(); R,C=w.shape
    L=llr_map(w)
    for it in range(2):
        L=L-np.median(L,1,keepdims=True); L=L-np.median(L,0,keepdims=True)
    L=np.clip(L,-6,6)
    h=int(round(R*np.sqrt(0.03))); ww=int(round(C*np.sqrt(0.03)))
    S=box(L,h,ww)
    i,j=np.unravel_index(int(S.argmax()),S.shape); sc=S[i,j]
    S2=S.copy(); S2[max(0,i-h):i+h, max(0,j-ww):j+ww]=-1e18; sc2=S2.max()
    i2,j2=np.unravel_index(int(S2.argmax()),S.shape)
    out[k]=[int(i),int(i+h),int(j),int(j+ww)]
    same = old[k]==out[k]
    print("%-26s rows %5d-%5d cols %4d-%4d score %7.0f 2nd %7.0f at (%d,%d) %s"%(k,i,i+h,j,j+ww,sc,sc2,i2,j2,"" if same else "CHANGED from %s"%old[k]),flush=True)
json.dump(out,open("blocks_v2.json","w"),indent=1)
