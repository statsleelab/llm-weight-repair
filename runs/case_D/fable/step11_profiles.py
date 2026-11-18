import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
from detlib import *
from step07_lib import loo2
sd = load_weights("damaged.safetensors")
np.set_printoptions(linewidth=250,precision=2,suppress=True)
cases={"h.10.attn.c_attn.weight":[(622,1983),(620,1843)],"h.10.attn.c_proj.weight":[(442,238)],"h.11.attn.c_proj.weight":[(0,62),(374,518)],
       "h.2.attn.c_attn.weight":[(458,65),(454,29)],"h.9.attn.c_attn.weight":[(144,1648),(144,1727)],"h.8.attn.c_attn.weight":[(42,1724)]}
for k,cands in cases.items():
    v=sd[k]; W=v.double().numpy(); n,m=W.shape; h,w=round(.1*n),round(.1*m); s2=W.var()
    Es=loo2(v,0.1); S=sum(norm1(np.minimum(norm1(E**2),8)) for E in Es)/2
    # shrink-pred covariance: P=W-E
    Cv=sum(norm1(np.abs(W))*0+ (W*(W-E)) for E in Es)/2
    Cv=Cv/np.sqrt((W**2).mean(1,keepdims=True)*(W**2).mean(0,keepdims=True))*np.sqrt((W**2).mean())
    print("=====",k,n,m,"box",h,w)
    for (i,j) in cands:
        bc=max(8,w//10); br=max(8,h//10)
        def prof(M,ax):
            if ax==1:
                p=M[i:i+h].mean(0); L=(m//bc)*bc; return p[:L].reshape(-1,bc).mean(1)
            p=M[:,j:j+w].mean(1); L=(n//br)*br; return p[:L].reshape(-1,br).mean(1)
        print(f"-- cand ({i},{j}); col bins of {bc} (block would span bins {j/bc:.1f}-{(j+w)/bc:.1f})")
        print(" W2/s2:",prof(W**2/s2,1)); print(" E2n  :",prof(S,1)); print(" cov  :",prof(Cv,1)*10)
        print(f"   row bins of {br} (block bins {i/br:.1f}-{(i+h)/br:.1f})")
        print(" W2/s2:",prof(W**2/s2,0)); print(" E2n  :",prof(S,0)); print(" cov  :",prof(Cv,0)*10)
