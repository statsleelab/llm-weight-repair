import torch, sys
torch.set_num_threads(1)
from gpt2_lib import *
ids=torch.load("dev_ids.pt").tolist()
sd=load_weights("damaged.safetensors")
x=torch.tensor([[EOT]+ids[:63]])
def glob(sd,T,cap=None):
    out={}
    for k,v in sd.items():
        if v.dim()!=2: out[k]=v; continue
        gm=((v-v.median()).abs().median()*1.4826).item()
        m=v.abs()>T*gm
        if cap: m&=v.abs()<cap*gm
        w=v.clone(); w[m]=0; out[k]=w
    return out
for T in [2.5,3.0,4.0]:
  for cap in [None,45]:
    r=glob(sd,T,cap)
    _,hid=forward(r,x,return_hidden=True)
    print(T,cap,"ppl",round(perplexity(r,ids,max_tokens=1024),2),"447 pos1+ L3",round(hid[3][0,1:,447].abs().mean().item(),1),"pos0",round(hid[3][0,0,447].item(),1),flush=True)
