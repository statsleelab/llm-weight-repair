import torch
torch.set_num_threads(1)
from gpt2_lib import *
from step09_wpe import repair_wpe
from step10_bayes import shrink
sd=load_weights("damaged.safetensors"); tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
new=dict(sd)
new["wpe.weight"]=repair_wpe(sd["wpe.weight"],10.0,1)
new["wte.weight"],pc=shrink(sd["wte.weight"],1.0)
print("wte entries pc>0.5:",(pc>0.5).sum().item())
save_weights(new,"repaired.safetensors")
r=load_weights("repaired.safetensors")
assert r.keys()==sd.keys() and all(r[k].shape==sd[k].shape for k in sd)
print("damaged ppl",perplexity(sd,ids,max_tokens=8192),"repaired ppl",perplexity(r,ids,max_tokens=8192))
