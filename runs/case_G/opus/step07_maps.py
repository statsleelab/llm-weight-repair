import torch; torch.set_num_threads(1)
import numpy as np, sys
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
keys=sys.argv[1:]
for k in keys:
    w=sd[k].numpy().astype(np.float64)
    R,C=w.shape
    s=w**2
    # iterative robust row/col scaling
    rs=np.ones(R); cs=np.ones(C)
    for _ in range(3):
        rs=np.median(s/cs[None],1); cs=np.median(s/rs[:,None],0)
    n=s/(rs[:,None]*cs[None])
    gr,gc=R//32,C//64
    t=n[:gr*32,:gc*64].reshape(32,gr,64,gc)
    m=t.mean((1,3)); 
    # kurtosis of raw values in tile
    x=w[:gr*32,:gc*64].reshape(32,gr,64,gc).transpose(0,2,1,3).reshape(32,64,-1)
    ku=((x-x.mean(-1,keepdims=True))**4).mean(-1)/x.var(-1)**2
    print(k, w.shape, "tile",gr,gc, "m median %.2f"%np.median(m))
    med=np.median(m); mk=np.median(ku)
    for i in range(32):
        print("".join("#" if a>1.5*med else ("+" if a>1.2*med else ("-" if a<0.7*med else ".")) for a in m[i]), " ", "".join("#" if q<3.3 else ("+" if q<0.8*mk else ".") for q in ku[i]))
