import torch, sys; torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import repair
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
print("dev tokens",len(ids))
sd=load_weights("damaged.safetensors")
print("damaged ppl", perplexity(sd, ids, max_tokens=4096))
for T in [6,8,12]:
    r,_=repair(sd,T=T,Twpe=T)
    print("T",T,"ppl",perplexity(r, ids, max_tokens=4096), flush=True)
