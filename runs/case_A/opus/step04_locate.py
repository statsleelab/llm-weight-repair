import torch, math; torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())[:256]
x=torch.tensor([[EOT]+ids])
logits,hid=forward(sd,x,return_hidden=True)
for i in range(1,13):
    d=(hid[i]-hid[i-1])[0,1:]  # skip first token
    m=d.abs().mean(0)
    top=m.topk(4)
    print(i, "delta mean-abs top dims", [(int(a),round(float(b),1)) for b,a in zip(top.values,top.indices)], "hid top", [(int(j),round(float(hid[i][0,1:,j].mean()),1)) for j in hid[i][0,1:].abs().mean(0).topk(3).indices])
