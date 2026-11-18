import sys, torch, time
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
print("dev tokens",len(ids))
for p in sys.argv[1:]:
    t=time.time(); sd=load_weights(p)
    nll,acc=nll_windows(sd,ids,max_tokens=8192)
    print(p, "ppl=%.3f acc=%.4f (%.0fs)"%(math.exp(nll),acc,time.time()-t))
    print(generate(sd,tok,"The capital of France is",15))
