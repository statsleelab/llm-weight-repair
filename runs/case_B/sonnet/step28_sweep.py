import torch,numpy as np,sys
torch.set_num_threads(1)
import step15_soft as S
from gpt2_lib import *
ids=S.ids[:1100]
for pi in [0.0105,0.05,0.25]:
    S.PI=pi
    for sc in [1.2]:
        S.SC=sc
        s2=S.repair(S.sd,verbose=False)
        print("PI",pi,"plain soft ppl %.4g"%perplexity(s2,ids,max_tokens=1024),flush=True)
