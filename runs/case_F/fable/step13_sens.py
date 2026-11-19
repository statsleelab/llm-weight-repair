# Sensitivity probe: re-corrupt one group of matrices of the current repair (as pseudo-truth) with the
# measured noise model, re-repair with the same procedure, and see how much dev ppl moves.
import torch, sys, math
torch.set_num_threads(1)
import replib
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
base=load_weights(sys.argv[1]); groups=sys.argv[2:]
g=torch.Generator().manual_seed(0)
print("base",math.exp(nll_windows(base,ids,max_tokens=1025)[0]))
for grp in groups:
    sd=dict(base)
    for k,v in base.items():
        if v.ndim==2 and grp in k:
            m=torch.rand(v.shape,generator=g)<0.03
            x=v+m*torch.randn(v.shape,generator=g)*10*v.std()
            sd[k]=replib.repair_matrix(k,x)
    print(grp,math.exp(nll_windows(sd,ids,max_tokens=1025)[0]),flush=True)
