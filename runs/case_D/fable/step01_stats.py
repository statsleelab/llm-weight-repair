import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.ndim!=2: continue
    a=v.numpy(); n=a.size
    fin=np.isfinite(a)
    s=a[fin].std(); 
    absa=np.abs(a[fin])
    q=np.quantile(absa[::7],[0.5,0.99,0.999,0.9999])
    print(f"{k:28s} {str(a.shape):14s} nonfin={(~fin).sum()} zeros={(a==0).mean():.5f} std={s:.4f} max={absa.max():.3g} q={np.round(q,3)} "
          f">6s={(absa>6*s).mean():.5f} >10s={(absa>10*s).mean():.5f} kurt={((a[fin]-a[fin].mean())**4).mean()/s**4:.1f}")
