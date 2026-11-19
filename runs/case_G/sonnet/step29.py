import torch,numpy as np,json
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("repaired.safetensors")
dm=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
bx=json.load(open("boxes3.json"))
def ridge_fill(W,a,bh,b,bw,lam):
    m=torch.ones(W.shape[0],dtype=torch.bool); m[a:a+bh]=False
    keep=torch.ones(W.shape[1],dtype=torch.bool); keep[b:b+bw]=False
    X=W[m][:,keep]; Y=W[m][:,b:b+bw]
    G=X.T@X; G+=lam*G.diagonal().mean()*torch.eye(G.shape[0])
    B=torch.linalg.solve(G,X.T@Y)
    return W[a:a+bh][:,keep]@B
n=0
for k,(a,bh,b,bw,_) in bx.items():
    if k=="wte.weight" or k.endswith("mlp.c_proj.weight"):
        W=dm[k].clone(); W[a:a+bh,b:b+bw]=0
        # train on clean rows only; features in box rows are clean (box cols excluded)
        sd[k][a:a+bh,b:b+bw]=ridge_fill(W,a,bh,b,bw,0.3)
        n+=1; print(k,flush=True)
print("ridge",nll_windows(sd,ids,window=512,max_tokens=4096))
save_weights(sd,"cand_ridge.safetensors")
