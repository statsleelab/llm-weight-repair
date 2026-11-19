import torch,numpy as np,json
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
bx=json.load(open("boxes2.json"))
w=sd["wte.weight"]
lw=torch.log(w.abs()+1e-2)
pred=lw.median(1,keepdim=True).values+lw.median(0,keepdim=True).values-lw.median()
def bc(a,b,bh,bw):
    x=lw[a:a+bh,b:b+bw].flatten();y=pred[a:a+bh,b:b+bw].flatten();x=x-x.mean();y=y-y.mean();return ((x*y).sum()/(x.norm()*y.norm())).item()
sc={b:bc(34395,b,8705,133) for b in range(612,630)}
b=min(sc,key=sc.get);print("wte col start",b)
bx["wte.weight"][2]=b
# row refine
sc={a:bc(a,b,8705,133) for a in range(34370,34425,3)};a=min(sc,key=sc.get);print("wte row start",a);bx["wte.weight"][0]=a
json.dump(bx,open("boxes3.json","w"))
# wpe interpolation
wp=sd["wpe.weight"]; r0,r1=124,303
a=wp[r0-1].clone(); c=wp[r1]
for i in range(r0,r1):
    t=(i-(r0-1))/(r1-(r0-1)); wp[i,591:724]=(a*(1-t)+c*t)[591:724]
for k,(a,bh,b,bw,_) in bx.items():
    sd[k][a:a+bh,b:b+bw]=0
print("repaired-zero",nll_windows(sd,ids,window=512,max_tokens=4096),flush=True)
save_weights(sd,"repaired.safetensors")
