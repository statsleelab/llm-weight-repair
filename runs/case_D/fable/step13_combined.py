import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
from detlib import *
from step07_lib import loo2
sd = load_weights("damaged.safetensors")
bE=json.load(open("blocks.json")); bG=json.load(open("blocks_gauss.json"))
final={}
for k,v in sd.items():
    if v.ndim!=2: continue
    W=v.double().numpy(); n,m=W.shape; h,w=round(.1*n),round(.1*m); N=h*w; s2=W.var()
    if k=="wpe.weight": final[k]=[151,253,59,136]; continue
    if n>5000:
        Wd=v.double(); G=Wd.T@Wd/n; Th=torch.linalg.inv(G+0.1*G.diagonal().mean()*torch.eye(m,dtype=torch.float64)); Es=[((Wd@Th)/Th.diagonal()[None,:]).numpy()]
    else: Es=loo2(v,0.1)
    S=sum(norm1(np.minimum(norm1(E**2),8)) for E in Es)/len(Es)
    F=boxsum(S,h,w)/N; med=np.median(F); sc=np.median(np.abs(F-med))*1.4826; zE=(F-med)/sc
    m1=boxsum(W,h,w)/N; m2=boxsum(W**2,h,w)/N; m4=boxsum(W**4,h,w)/N; m3=boxsum(W**3,h,w)/N
    Q=((m2/s2-1)/np.sqrt(2/N))**2+(m1/np.sqrt(s2/N))**2+((m4/m2**2-3)/np.sqrt(24/N))**2+((m3/m2**1.5)/np.sqrt(6/N))**2
    Cm=zE-Q/2
    i,j=np.unravel_index(Cm.argmax(),Cm.shape)
    C2=Cm.copy(); C2[max(0,i-h//2):i+h//2,max(0,j-w//2):j+w//2]=-1e18
    i2,j2=np.unravel_index(C2.argmax(),C2.shape)
    e=bE[k]; g=bG[k]
    print(f"{k:24s} comb@({i},{j}) C={Cm[i,j]:.1f} zE={zE[i,j]:.1f} Q={Q[i,j]:.1f} | runner@({i2},{j2}) C={C2[i2,j2]:.1f} | E2@({e[0]},{e[2]}) zE={zE[e[0],e[2]]:.1f} Q={Q[e[0],e[2]]:.1f} | G@({g[0]},{g[2]}) zE={zE[g[0],g[2]]:.1f} Q={Q[g[0],g[2]]:.1f}",flush=True)
    final[k]=[int(i),int(i+h),int(j),int(j+w)]
json.dump(final,open("blocks_comb.json","w"),indent=0)
