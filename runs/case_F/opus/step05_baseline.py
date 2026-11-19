import torch, sys, time
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
print("dev tokens",len(ids))
torch.save(torch.tensor(ids),"dev_ids.pt")
sd=load_weights("damaged.safetensors")
t=time.time(); print("damaged ppl(4096)",perplexity(sd,ids,max_tokens=4096), time.time()-t)
print(generate(sd,tok,"The capital of France is",15))
