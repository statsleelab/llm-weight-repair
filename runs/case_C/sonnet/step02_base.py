import torch,time
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k in ["wte.weight","wpe.weight"]:
    v=sd[k]; print(k,v.shape,v.std().item(),(v==0).sum().item(),(v==0).float().mean().item())
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
print(len(ids))
t=time.time()
print("ppl",perplexity(sd,ids,max_tokens=4096),time.time()-t)
