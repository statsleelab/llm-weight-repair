import torch
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
ids=torch.load("dev_ids.pt").tolist()
sd=load_weights("damaged.safetensors")
r=repair(sd,T=4,protect=True)
x=torch.tensor([[EOT]+ids[:63]])
def chk(name,m):
    _,hid=forward(m,x,return_hidden=True)
    print(name,"ppl",round(perplexity(m,ids,max_tokens=1024),2),"L1/L3 dim447 pos1+",[round(hid[l][0,1:,447].abs().mean().item(),1) for l in (1,2,3)], "pos0",[round(hid[l][0,0,447].item(),1) for l in (1,2,3)],flush=True)
chk("base",r)
m=dict(r); m["wpe.weight"]=sd["wpe.weight"]; chk("wpe damaged",m)
for L in range(3):
    for nm in ["attn.c_attn","attn.c_proj","mlp.c_fc","mlp.c_proj"]:
        k=f"h.{L}.{nm}.weight"
        m=dict(r); m[k]=sd[k]; chk("unrepaired "+k,m)
