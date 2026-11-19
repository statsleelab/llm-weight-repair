import torch, json, sys, time, math
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read()); print("dev tokens",len(ids))
N=int(sys.argv[2]) if len(sys.argv)>2 else 4096
B=json.load(open("blocks_final.json"))
which=sys.argv[1]
t=time.time()
if which=="damaged": sd=load_weights("damaged.safetensors")
elif which=="zero":
    sd=load_weights("damaged.safetensors")
    for k,(r0,r1,c0,c1) in B.items(): sd[k][r0:r1,c0:c1]=0
else: sd=load_weights(which)
nll,acc=nll_windows(sd,ids,max_tokens=N)
print(f"{which}: nll={nll:.4f} ppl={math.exp(nll):.3f} acc={acc:.4f} [{time.time()-t:.0f}s]")
