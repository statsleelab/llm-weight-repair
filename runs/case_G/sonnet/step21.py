import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
from blk import *
sd=load_weights("damaged.safetensors")
c=tilecorr(sd["wte.weight"],512,16)
print(c.shape)
np.set_printoptions(linewidth=250,precision=1)
print("min row-tile",c.min(1).values.numpy()[60:95])
print("per col tile at row-tile 80",c[80].numpy())
