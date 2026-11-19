# choose between a handful of candidate repairs per matrix group (dev ppl, 2048 tokens)
import torch, sys, math
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
base=load_weights(sys.argv[1]); 
def ev(sd): return math.exp(nll_windows(sd,ids,max_tokens=2049)[0])
print("base",ev(base),flush=True)
groups=sys.argv[3].split(",")
for alt in sys.argv[2].split(","):
    o=load_weights(alt)
    for grp in groups:
        sd={k:(o[k] if grp in k else v) for k,v in base.items()}
        print(alt,grp,ev(sd),flush=True)
