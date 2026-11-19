import torch
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer")
def devids(n=4096):
    return tok.encode(open("dev.txt",encoding="utf-8").read())[:n]
