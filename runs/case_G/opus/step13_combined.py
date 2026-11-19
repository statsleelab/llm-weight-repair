import torch; torch.set_num_threads(1)
import numpy as np, json, sys
from gpt2_lib import *
from detect_lib import llr_map
sd=load_weights("damaged.safetensors")
def box(L,h,w):
    c=np.zeros((L.shape[0]+1,L.shape[1]+1)); c[1:,1:]=L.cumsum(0).cumsum(1)
    return c[h:,w:]-c[:-h,w:]-c[:-h,w:]*0-c[h:,:-w]+c[:-h,:-w]
def best2(S,h,ww):
    i,j=np.unravel_index(int(S.argmax()),S.shape); sc=S[i,j]
    S2=S.copy(); S2[max(0,i-h):i+h, max(0,j-ww):j+ww]=-1e18
    i2,j2=np.unravel_index(int(S2.argmax()),S.shape)
    return (int(i),int(j),sc),(int(i2),int(j2),S2[i2,j2])
def zs(S): return (S-np.median(S))/(np.median(np.abs(S-np.median(S)))+1e-9)
keys=[k for k,v in sd.items() if v.dim()==2 and k!="wte.weight"]
old=json.load(open("blocks.json")); out={"wte.weight":old["wte.weight"]}
for k in keys:
    w=sd[k].numpy().astype(np.float64); R,C=w.shape
    h=int(round(R*np.sqrt(0.03))); ww=int(round(C*np.sqrt(0.03)))
    U,S,Vt=np.linalg.svd(w,full_matrices=False)
    lr=(U[:,:64]*S[:64])@Vt[:64]
    a=w*lr
    a=a/ (np.median(np.abs(a),0,keepdims=True)+1e-12)
    a=a/ (np.median(np.abs(a),1,keepdims=True)+1e-12)
    E=np.clip(-a,-10,10); E=E-np.median(E)
    L=llr_map(w)
    SE=zs(box(E,h,ww)); SL=zs(box(L,h,ww))
    pe,_=best2(SE,h,ww); pl,_=best2(SL,h,ww)
    c,c2=best2(SE+SL,h,ww)
    out[k]=[c[0],c[0]+h,c[1],c[1]+ww]
    print("%-24s svd(%d,%d) llr(%d,%d) comb(%d,%d) z=%.0f  2nd(%d,%d) z=%.0f %s"%(k,pe[0],pe[1],pl[0],pl[1],c[0],c[1],c[2],c2[0],c2[1],c2[2], "WEAK" if c2[2]>0.7*c[2] else ""),flush=True)
json.dump(out,open("blocks_comb.json","w"),indent=1)
