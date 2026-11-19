import torch
torch.set_num_threads(1)
from gpt2_lib import *
import repair3 as R
ids=torch.load("dev_ids.pt").tolist()
sd=load_weights("damaged.safetensors")
r=R.repair(sd,mode="soft",nu=4.0,line_adapt=True)
assert set(r)==set(sd) and all(r[k].shape==sd[k].shape for k in sd)
save_weights(r,"repaired.safetensors")
r2=load_weights("repaired.safetensors")
print("repaired ppl(4096)",perplexity(r2,ids,max_tokens=4096))
tok=Tokenizer("tokenizer")
print(generate(r2,tok,"The capital of France is",12))
