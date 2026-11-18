import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, nll_windows
sd = load_weights("damaged.safetensors")
tok = Tokenizer("tokenizer")
ids = tok.encode(open("dev.txt").read())
nll, acc = nll_windows(sd, ids[:2048], batch=4)
print("damaged nll (2048 tok)", nll, "acc", acc)
wte = sd["wte.weight"].double()
med = wte.median(dim=0).values
mad = (wte-med).abs().median(dim=0).values*1.4826
order = torch.argsort(mad)
print("columns with smallest robust sd:")
for j in order[:20].tolist():
    print(j, "median", round(med[j].item(),4), "mad", round(mad[j].item(),5), "std", round(wte[:,j].std().item(),4), "mean", round(wte[:,j].mean().item(),4))
print("median of mad over columns", mad.median().item())
print("columns with largest |median|:")
for j in torch.argsort(-med.abs())[:8].tolist():
    print(j, round(med[j].item(),4), round(mad[j].item(),5))
