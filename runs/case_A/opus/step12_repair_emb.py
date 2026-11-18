import torch, time, math; torch.set_num_threads(1)
from gpt2_lib import *
from imputers import *
sd = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
new=dict(sd)
new["wte.weight"]=gauss_impute(sd["wte.weight"],sd["wte.weight"]==0,lam=0.003,rounds=3)
new["wpe.weight"]=gauss_impute(sd["wpe.weight"],sd["wpe.weight"]==0,lam=0.01,rounds=3)
save_weights({k:new[k] for k in ["wte.weight","wpe.weight"]},"emb_fixed.safetensors")
for name,s in [("wte+wpe fixed",new)]:
    nll,acc=nll_windows(s,ids,max_tokens=8192); print(name,"ppl",math.exp(nll),"acc",acc,flush=True)
print(generate(new,tok,"The capital of France is",15))
