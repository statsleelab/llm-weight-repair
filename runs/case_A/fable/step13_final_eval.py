import torch, math
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
N=8192
for f in ["cand_v1.safetensors","repaired.safetensors"]:
    sd=load_weights(f); nll,acc=nll_windows(sd,ids,max_tokens=N)
    print(f,"tokens",N,"ppl %.3f acc %.4f"%(math.exp(nll),acc),flush=True)
print(repr(generate(sd,tok,"The capital of France is",12)))
print(sentence_logprob(sd,tok,["The keys to the cabinet are on the table.","The keys to the cabinet is on the table."]))
# sensitivity probe: how much does a *fresh, unrepaired* 1% zeroing of block weights / wpe cost?
g=torch.Generator().manual_seed(5)
s2={k:(v*(torch.rand(v.shape,generator=g)>=0.01) if (v.ndim==2 and k!="wte.weight") else v) for k,v in sd.items()}
print("repaired + fresh unrepaired 1%% zeros in all h.*/wpe matrices: ppl %.3f"%math.exp(nll_windows(s2,ids,max_tokens=N)[0]))
