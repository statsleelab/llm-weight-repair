import torch,time
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
x=torch.randint(0,50000,(1,128))
t=time.time(); forward(sd,x); print("128 tok fwd",time.time()-t)
