import torch, sys
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
ids=torch.load("dev_ids.pt").tolist()
sd=load_weights("damaged.safetensors")
for arg in sys.argv[1:]:
    T,prot=arg.split(","); T=float(T); prot=prot=="1"
    r=repair(sd,T=T,protect=prot,verbose=(arg==sys.argv[1]))
    x=torch.tensor([[EOT]+ids[:63]])
    _,hid=forward(r,x,return_hidden=True)
    print(arg,"ppl",perplexity(r,ids,max_tokens=4096),"pos0 norms",[round(h[0,0].norm().item()) for h in hid[::3]],flush=True)
