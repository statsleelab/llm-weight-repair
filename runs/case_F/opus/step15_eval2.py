import torch, sys
torch.set_num_threads(1)
from gpt2_lib import *
import repair2 as R
ids=torch.load("dev_ids.pt").tolist()
sd=load_weights("damaged.safetensors")
x=torch.tensor([[EOT]+ids[:63]])
for arg in sys.argv[1:]:
    mode,nu,la=arg.split(",")
    r=R.repair(sd,mode=mode,nu=float(nu),line_adapt=la=="1")
    _,hid=forward(r,x,return_hidden=True)
    print(arg,"ppl",round(perplexity(r,ids,max_tokens=1024),2),"dim447 pos1+ L3",round(hid[3][0,1:,447].abs().mean().item(),1),"pos0",round(hid[3][0,0,447].item(),1),flush=True)
