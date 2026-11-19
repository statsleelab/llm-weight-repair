import torch; torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import repair
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())[:256]
sd=load_weights("damaged.safetensors")
r,_=repair(sd,T=8)
for name,s in [("damaged",sd),("rep8",r)]:
    lg,h=forward(s,torch.tensor([[EOT]+ids]),return_hidden=True)
    print(name,[round(x[0].norm(dim=-1).median().item(),1) for x in h])
    print("  maxabs",[round(x[0].abs().max().item(),1) for x in h])
    print("  logits maxabs", lg.abs().max().item(), "argmax top dims", h[-1][0].abs().mean(0).topk(5).indices.tolist())
    print(" gen:", repr(generate(s,tok,"The capital of France is",10)))
