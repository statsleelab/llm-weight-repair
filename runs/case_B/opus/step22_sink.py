import torch; torch.set_num_threads(1)
from gpt2_lib import *
R=torch.load("/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/r5g.pt")
D=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())[:64]
x=torch.tensor([[EOT]+ids])
for name,sd in [("repaired",R),("damaged",D)]:
    _,hid=forward(sd,x,return_hidden=True)
    print(name,"pos0 norm per layer",[round(h[0,0].norm().item()) for h in hid])
    print(name,"pos5 norm per layer",[round(h[0,5].norm().item()) for h in hid])
    print(name,"pos0 top dims L4", hid[4][0,0].abs().topk(3))
# entries changed in column 3034 of h.2.mlp.c_fc and row 3034 of c_proj
for k,ax,idx in [("h.2.mlp.c_fc.weight",1,3034),("h.3.mlp.c_fc.weight",1,1751)]:
    d=D[k].select(ax,idx); r=R[k].select(ax,idx); ch=(d!=r).nonzero().flatten()
    print(k,idx,"changed",[(i,round(d[i].item(),2),round(r[i].item(),2)) for i in ch.tolist()][:12], "col absmax", d.abs().topk(6).values.tolist())
