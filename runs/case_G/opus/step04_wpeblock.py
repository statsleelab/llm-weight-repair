import torch; torch.set_num_threads(1)
import numpy as np
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wpe.weight"].numpy()
P=w.shape[0]
pad=np.pad(w,((2,2),(0,0)),mode='edge')
nb=np.stack([pad[0:P],pad[1:P+1],pad[3:P+3],pad[4:P+4]])
pred=np.median(nb,0)
spread=np.median(np.abs(nb-pred),0)+1e-3
z=np.abs(w-pred)/spread
B=16
t=(z>5).reshape(P//B,B,768//B,B).mean((1,3))
np.set_printoptions(linewidth=250)
for i in range(0,24):
    print("%4d "%(i*B)+"".join("#" if x>0.3 else ("+" if x>0.1 else ".") for x in t[i]))
# find exact boundaries
rows=np.where((z>5)[:,560:740].mean(1)>0.2)[0]; print("rows",rows.min(),rows.max(),len(rows))
cols=np.where((z>5)[120:340].mean(0)>0.2)[0]; print("cols",cols.min(),cols.max(),len(cols))
