import torch, sys, math
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
N=int(sys.argv[2]) if len(sys.argv)>2 else 4096
sd=load_weights(sys.argv[1])
nll,acc=nll_windows(sd,ids,max_tokens=N)
print(sys.argv[1],"tokens",N,"ppl %.3f nll %.4f acc %.4f"%(math.exp(nll),nll,acc))
print(repr(generate(sd,tok,"The capital of France is",12)))
