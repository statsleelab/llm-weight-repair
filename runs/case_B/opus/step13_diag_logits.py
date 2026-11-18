import torch; torch.set_num_threads(1)
import torch.nn.functional as F
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())[:512]
sd=dict(D)
for k,v in D.items():
    if v.dim()!=2: continue
    sd[k]=repair_wpe(v,6)[0] if k=="wpe.weight" else repair2(v,5)[0]
torch.save(sd,"/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/r5.pt")
x=torch.tensor([[EOT]+ids])
logits,hid=forward(sd,x,return_hidden=True)
h=F.layer_norm(hid[-1],(768,),sd["ln_f.weight"],sd["ln_f.bias"],1e-5)[0]
xs=h.abs().mean(0); print("final x top dims", xs.topk(8))
for i,hh in enumerate(hid): print(i,"norm %.1f"%hh[0,1:].norm(dim=-1).mean().item(), "top dims", hh[0,1:].abs().mean(0).topk(4).indices.tolist(), [round(t,1) for t in hh[0,1:].abs().mean(0).topk(4).values.tolist()])
ml=logits[0].mean(0); top=ml.topk(10)
print("tokens with highest mean logit:", [(tok.decode([i]),round(v,1)) for v,i in zip(top.values.tolist(),top.indices.tolist())])
print("median of mean logits", ml.median().item())
W=sd["wte.weight"]; hm=h.mean(0)
for t in top.indices[:4].tolist():
    c=W[t]*hm; tc=c.abs().topk(5)
    print(tok.decode([t]), [(j, round(W[t,j].item(),3), round(c[j].item(),1)) for j in tc.indices.tolist()])
