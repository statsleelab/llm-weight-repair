import torch; torch.set_num_threads(1)
from gpt2_lib import *
D=load_weights("damaged.safetensors"); R=load_weights("repaired.safetensors")
assert D.keys()==R.keys() and all(D[k].shape==R[k].shape and R[k].dtype==torch.float32 for k in D)
assert all(torch.equal(D[k],R[k]) for k in D if D[k].dim()==1)
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
for n,sd in [("damaged",D),("repaired",R)]:
    nll,acc=nll_windows(sd,ids,max_tokens=8193); print(n,"ppl8192 %.4g acc %.3f"%(math.exp(nll),acc))
print(generate(R,tok,"The capital of France is",12))
