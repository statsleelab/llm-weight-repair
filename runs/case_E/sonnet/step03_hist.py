import torch
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
v=sd["wte.weight"].flatten()
edges=[0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,3.5,4,5,7]
a=v.abs()
for lo,hi in zip(edges[:-1],edges[1:]): print(lo,hi,((a>=lo)&(a<hi)).sum().item())
v=sd["wpe.weight"]
print("wpe per-row std first rows",v.std(1)[:5],v.std(1)[500:505])
print("wpe row std quantiles",v.std(1).quantile(torch.tensor([0,.1,.5,.9,1.])))
print("wpe col0..: ", v[:3,:6])
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())
print(len(ids))
print("damaged ppl",perplexity(sd,ids,max_tokens=4096))
print(generate(sd,tok,"The capital of France is",15))
