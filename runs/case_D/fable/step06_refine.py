import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
from detlib import *
sd = load_weights("damaged.safetensors")
blocks={}
for k,v in sd.items():
    if v.ndim!=2: continue
    n,m=v.shape; h,w=round(.1*n),round(.1*m)
    if k=="wpe.weight":
        W=v.numpy(); med=np.median(np.stack([np.roll(W,d,0) for d in (-3,-2,-1,1,2,3)]),0)
        S=norm1((W-med)**2+1e-12)
    else:
        S=norm1(loo_resid(v).numpy()**2)
    F=boxsum(S,h,w)/(h*w)
    # top distinct candidates
    cands=[]
    F2=F.copy()
    for _ in range(3):
        i,j=np.unravel_index(F2.argmax(),F2.shape)
        cands.append((int(i),int(j),float(F2[i,j])))
        F2[max(0,i-h):i+h,max(0,j-w):j+w]=-1
    out=[]
    for i,j,f in cands[:2]:
        r0,r1,c0,c1,rin,rout,rsd=refine(S,i,i+h,j,j+w)
        out.append((r0,r1,c0,c1,(rin-rout)/rsd))
    blocks[k]=[int(x) for x in out[0][:4]]
    print(f"{k:26s} {n}x{m} exp {h}x{w} cand1 rows {out[0][0]}-{out[0][1]} ({out[0][1]-out[0][0]}) cols {out[0][2]}-{out[0][3]} ({out[0][3]-out[0][2]}) contrast {out[0][4]:.1f} peak {cands[0][2]:.3f} || cand2 {out[1][:4]} contrast {out[1][4]:.1f} peak {cands[1][2]:.3f}",flush=True)
json.dump(blocks,open("blocks.json","w"),indent=0)
