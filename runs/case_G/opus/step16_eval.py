import torch; torch.set_num_threads(1)
import sys
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
n=int(sys.argv[2]) if len(sys.argv)>2 else 4096
sd=load_weights(sys.argv[1])
nll,acc=nll_windows(sd,ids,window=512,max_tokens=n+1)
print(sys.argv[1],"tokens",n,"ppl %.3f acc %.4f"%(math.exp(nll),acc))
