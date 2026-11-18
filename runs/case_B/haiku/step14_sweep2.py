import torch, math, sys
torch.set_num_threads(1)
from gpt2_lib import load_weights, forward, Tokenizer, nll_windows
from step07_repair import detect_and_fill
from step13_sweep import build, sd0, ids
configs = [(0,1.5,None),(0,1.0,None),(8,1.5,None),(12,1.5,None),(0,1.5,4.0)]
for Tlin,Twte,Twpe in configs:
    sd = build(Tlin,Twte,Twpe)
    nll, acc = nll_windows(sd, ids[:4096])
    print(f"Tlin={Tlin} Twte={Twte} Twpe={Twpe} nll4096={nll:.3f} ppl={math.exp(nll):.2f} top1={acc:.4f}", flush=True)
