import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
from detlib import boxsum
sd = load_weights("damaged.safetensors")
old=json.load(open("blocks.json")) if True else {}
new={}
for k,v in sd.items():
    if v.ndim!=2: continue
    W=v.double().numpy(); n,m=W.shape; h,w=round(.1*n),round(.1*m); N=h*w
    s2=W.var()
    m1=boxsum(W,h,w)/N; m2=boxsum(W**2,h,w)/N; m4=boxsum(W**4,h,w)/N; m3=boxsum(W**3,h,w)/N
    zv=(m2/s2-1)/np.sqrt(2/N); zm=m1/np.sqrt(s2/N)
    zk=(m4/m2**2-3)/np.sqrt(24/N); zs=(m3/m2**1.5)/np.sqrt(6/N)
    Q=zv**2+zk**2+zm**2+zs**2
    i,j=np.unravel_index(Q.argmin(),Q.shape)
    Q2=Q.copy(); Q2[max(0,i-h//2):i+h//2,max(0,j-w//2):j+w//2]=1e18
    i2,j2=np.unravel_index(Q2.argmin(),Q2.shape)
    o=old.get(k,[0,0,0,0])
    flag="" if (abs(o[0]-i)<=1 and abs(o[2]-j)<=1) else "  <-- differs from E2 "+str((o[0],o[2]))
    print(f"{k:26s} best@({i},{j}) Q={Q[i,j]:.1f} [zv {zv[i,j]:.1f} zk {zk[i,j]:.1f} zm {zm[i,j]:.1f} zs {zs[i,j]:.1f}] runner@({i2},{j2}) Q={Q2[i2,j2]:.1f} medianQ={np.median(Q):.0f}{flag}",flush=True)
    new[k]=[int(i),int(i+h),int(j),int(j+w)]
json.dump(new,open("blocks_gauss.json","w"),indent=0)
