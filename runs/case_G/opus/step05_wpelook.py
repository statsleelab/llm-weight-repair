import torch; torch.set_num_threads(1)
import numpy as np
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd["wpe.weight"].numpy()
np.set_printoptions(linewidth=200,precision=3,suppress=True)
# roughness per row/col: abs second difference along position
d2=np.abs(w[2:]-2*w[1:-1]+w[:-2])
r=d2[100:400].T  # cols x pos
colrough=d2[300:1000].mean(0)
inside=d2[140:280].mean(0)/ (colrough+1e-6)
print("col ratio > 3:", np.where(inside>3)[0])
c=np.where(inside>3)[0]
rowr=d2[:, c].mean(1)/ d2[:, :550].mean(1)
print("rows ratio>3:", np.where(rowr>3)[0]+1)
print(w[118:135,650])
print(w[295:310,650])
