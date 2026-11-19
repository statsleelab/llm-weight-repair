import torch; torch.set_num_threads(1)
import numpy as np, json, sys
from gpt2_lib import *
from detect_lib import llr_map
sd=load_weights("damaged.safetensors")
def box(L,h,w):
    c=np.zeros((L.shape[0]+1,L.shape[1]+1)); c[1:,1:]=L.cumsum(0).cumsum(1)
    return c[h:,w:]-c[:-h,w:]-c[h:,:-w]+c[:-h,:-w]
def best2(S,h,ww):
    i,j=np.unravel_index(int(S.argmax()),S.shape); sc=S[i,j]
    S2=S.copy(); S2[max(0,i-h):i+h, max(0,j-ww):j+ww]=-1e18
    i2,j2=np.unravel_index(int(S2.argmax()),S.shape)
    return (int(i),int(j),sc),(int(i2),int(j2),S2[i2,j2])
keys=sys.argv[2:]; K=int(sys.argv[1])
for k in keys:
    w=sd[k].numpy().astype(np.float64); R,C=w.shape
    h=int(round(R*np.sqrt(0.03))); ww=int(round(C*np.sqrt(0.03)))
    U,S,Vt=np.linalg.svd(w,full_matrices=False)
    lr=(U[:,:K]*S[:K])@Vt[:K]
    E=-(w*lr)/np.mean(np.abs(w*lr))
    E=E-E.mean()
    a,b=best2(box(E,h,ww),h,ww)
    print("%-24s SVD k=%d best (%d,%d) %.0f  2nd (%d,%d) %.0f  ratio %.2f"%(k,K,*a,*b,b[2]/a[2]))
