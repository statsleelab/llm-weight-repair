import os,sys,time
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
from imp import impute
lo,hi=int(sys.argv[1]),int(sys.argv[2])   # layer range; 12 => embeddings
sd=load_weights("damaged.safetensors")
import os.path
out=dict(sd) if not os.path.exists("repaired_tmp.pt") else torch.load("repaired_tmp.pt")
cfg={"attn.c_attn.weight":(1,0.3),"attn.c_proj.weight":(0,0.3),"mlp.c_fc.weight":(1,0.6),"mlp.c_proj.weight":(0,0.3)}
def fix(name,axis,lam):
    t=time.time()
    W=sd[name].numpy().astype(np.float64); Z=W==0
    X=W if axis==0 else W.T; M=Z if axis==0 else Z.T
    Xi=impute(X,M,lam=lam,iters=2)
    R=Xi if axis==0 else Xi.T
    out[name]=torch.tensor(R,dtype=torch.float32)
    print(name,"fixed",int(Z.sum()),f"{time.time()-t:.0f}s",flush=True)
for L in range(lo,hi):
    if L==12:
        fix("wpe.weight",0,0.3); fix("wte.weight",0,0.3)
    else:
        for k,(a,l) in cfg.items(): fix(f"h.{L}.{k}",a,l)
torch.save(out,"repaired_tmp.pt")
