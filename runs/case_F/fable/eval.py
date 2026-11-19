import torch, sys, math
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
n=int(sys.argv[2]) if len(sys.argv)>2 else 4096
sd=load_weights(sys.argv[1]); nll,acc=nll_windows(sd,ids,max_tokens=n+1)
print(sys.argv[1],"tokens",len(ids),"ppl",math.exp(nll),"acc",acc)
