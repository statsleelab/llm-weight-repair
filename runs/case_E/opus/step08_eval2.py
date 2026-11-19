import torch; torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib2 import repair
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
sd=load_weights("damaged.safetensors")
r,_=repair(sd,T=8,verbose=True)
lg,h=forward(r,torch.tensor([[EOT]+ids[:128]]),return_hidden=True)
print("maxabs",[round(x[0].abs().max().item(),1) for x in h])
print("ppl T8", perplexity(r, ids, max_tokens=2048))
print(generate(r,tok,"The capital of France is",10))
