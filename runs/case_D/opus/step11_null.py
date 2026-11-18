import torch, numpy as np, sys
torch.set_num_threads(1)
import torch.nn.functional as F
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
K=16
def nullmap(v):
    g=v.std()
    x=(v/g)
    n,m=x.shape; n2,m2=n//K*K,m//K*K
    t=x[:n2,:m2].reshape(n2//K,K,m2//K,K).permute(0,2,1,3).reshape(n2//K,m2//K,K*K)
    var=t.var(-1); mean=t.mean(-1)
    # chi2 statistics: deviation of var from 1 (Gaussian null var of sample var ~ 2/(N-1))
    zvar=(var-1)/np.sqrt(2/(K*K-1))
    zmean=mean*np.sqrt(K*K)
    # row-var heterogeneity within window
    tr=x[:n2,:m2].reshape(n2//K,K,m2//K,K)
    rv=(tr**2).mean(3)  # (n/K,K,m/K)
    cv=(tr**2).mean(1)  # (n/K,m/K,K)
    het=rv.var(1)+cv.var(2)  # under null ~ 2*(2/K)
    stat=zvar**2+zmean**2+ (het/(4/K))
    return stat
for k in sys.argv[1:]:
    s=nullmap(sd[k])
    lo=(s<np.quantile(s.numpy(),0.02))
    print(k,"stat median",s.median().item())
    # find rows/cols with many low-stat windows
    rp=lo.float().mean(1); cp=lo.float().mean(0)
    print(" rows blocks:",''.join('#' if a>0.3 else ('+' if a>0.1 else '.') for a in rp.tolist()))
    print(" cols blocks:",''.join('#' if a>0.3 else ('+' if a>0.1 else '.') for a in cp.tolist()))
