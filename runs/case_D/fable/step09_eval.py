import torch, sys, math
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt",encoding="utf-8").read())
off=int(sys.argv[1]) if sys.argv[1].isdigit() else 0
files=sys.argv[2:] if sys.argv[1].isdigit() else sys.argv[1:]
for f in files:
    sd=load_weights(f); nll,acc=nll_windows(sd,ids[off:],window=512,max_tokens=6144)
    print(f"{f}: tokens[{off}:{off+6144}] nll {nll:.4f} ppl {math.exp(nll):.3f} acc {acc:.4f}",flush=True)
