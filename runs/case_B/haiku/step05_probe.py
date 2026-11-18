import torch, time
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, perplexity, nll_windows
sd = load_weights("damaged.safetensors")
wpe = sd["wpe.weight"].abs().flatten()
print("wpe |w| quantiles", [round(v,4) for v in torch.quantile(wpe[::7].double(), torch.tensor([0.1,0.25,0.5,0.75,0.9,0.95,0.98,0.99,0.995],dtype=torch.float64)).tolist()])
print("wpe rows 0..3 abs median", sd["wpe.weight"][:4].abs().median(dim=1).values.tolist(), "row 1023", sd["wpe.weight"][-1].abs().median().item())
wte = sd["wte.weight"]
c36 = wte[:,36]
print("wte col36 quantiles", [round(v,4) for v in torch.quantile(c36.double(), torch.tensor([0.0,0.01,0.1,0.5,0.9,0.99,1.0],dtype=torch.float64)).tolist()])
print("wte col36 unique-ish count", c36.unique().numel())
tok = Tokenizer("tokenizer")
ids = tok.encode(open("dev.txt").read())
t=time.time()
nll, acc = nll_windows(sd, ids)
print("damaged dev ppl", torch.exp(torch.tensor(nll)).item(), "acc", acc, "time", time.time()-t)
