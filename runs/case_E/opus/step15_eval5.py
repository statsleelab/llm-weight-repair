import torch; torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib5 import repair
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
sd=load_weights("damaged.safetensors")
r,_=repair(sd,T=6,verbose=True)
lg,h=forward(r,torch.tensor([[EOT]+ids[:128]]),return_hidden=True)
print("maxabs",[round(x[0].abs().max().item(),1) for x in h])
print(generate(r,tok,"The capital of France is",12))
for T in [5,6,8]:
    r,_=repair(sd,T=T); print("T",T,"ppl",perplexity(r, ids, max_tokens=2048),flush=True)
