import torch, sys, math
torch.set_num_threads(1)
from gpt2_lib import *
import torch.nn.functional as F
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())[:512]
sd=load_weights(sys.argv[1])
x=torch.tensor([[EOT]+ids])
lg,h=forward(sd,x[:,:-1],return_hidden=True)
nll=F.cross_entropy(lg[0],x[0,1:],reduction="none")
print("nll by pos block of 64:",[round(nll[i:i+64].mean().item(),2) for i in range(0,512,64)])
top=lg[0].argmax(-1); vals,cnt=torch.unique(top,return_counts=True)
o=cnt.argsort(descending=True)[:12]
print([(tok.decode([vals[i]]),cnt[i].item()) for i in o])
for i,xx in enumerate(h): print(i,"norm pos0 %.1f  others mean %.1f max %.1f"%(xx[0,0].norm(),xx[0,1:].norm(dim=-1).mean(),xx[0,1:].norm(dim=-1).max()), "top dims",xx[0,1:].abs().mean(0).topk(4).indices.tolist(),[round(a,1) for a in xx[0,1:].abs().mean(0).topk(4).values.tolist()])
print(repr(generate(sd,tok,"The capital of France is",12)))
# logit bias: mean logit per token over positions - which tokens are favoured overall
ml=lg[0].mean(0); t=ml.topk(10)
print("highest mean-logit tokens",[(tok.decode([i]),round(v,1)) for i,v in zip(t.indices.tolist(),t.values.tolist())], "median",ml.median().item(), "target mean",lg[0,torch.arange(512),x[0,1:]].mean().item())
