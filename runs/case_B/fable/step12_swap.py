import torch, time, math
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, perplexity
a=load_weights("cand_eb4.safetensors"); b=load_weights("cand_eb5.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
t=time.time()
print("eb4 2k:",perplexity(a,ids,max_tokens=2048), time.time()-t)
for g in ["wte","wpe","attn.c_attn","attn.c_proj","mlp.c_fc","mlp.c_proj"]:
    m={k:(b[k] if g in k else a[k]) for k in a}
    print("eb4 with",g,"from eb5:",perplexity(m,ids,max_tokens=2048),flush=True)
    d=[( (b[k]-a[k]).abs().max().item(),k) for k in a if g in k]
    print("    max |diff|",max(d))
