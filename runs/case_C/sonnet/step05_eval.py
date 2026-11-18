import torch
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
sd=torch.load("repaired_tmp.pt")
print("ppl",perplexity(sd,ids,max_tokens=8192))
print(generate(sd,tok,"The capital of France is",15))
