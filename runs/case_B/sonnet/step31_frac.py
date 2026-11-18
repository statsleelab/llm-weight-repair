import torch,numpy as np,sys
torch.set_num_threads(1)
from gpt2_lib import *
import step15_soft as S
from step17_heavy import repair2
from step30_sink import sink
ids=S.ids[:1100]
for frac in [0.02,0.012]:
    s2=repair2(S.sd,frac=frac,absk=5.0)
    print("frac",frac,"sink",sink(s2),"ppl %.4g"%perplexity(s2,ids,max_tokens=1024),flush=True)
    save_weights(s2,f"cand_frac{frac}.safetensors")
