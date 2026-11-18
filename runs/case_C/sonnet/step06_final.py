import torch
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
d=load_weights("damaged.safetensors"); r=torch.load("repaired_tmp.pt")
print("damaged ppl 8192:",perplexity(d,ids,max_tokens=8192))
print("repaired ppl 8192:",perplexity(r,ids,max_tokens=8192))
print("repaired ppl full:",perplexity(r,ids))
save_weights(r,"repaired.safetensors")
r2=load_weights("repaired.safetensors"); print(len(r2),all(r2[k].shape==d[k].shape for k in d))
