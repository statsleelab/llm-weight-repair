import torch,numpy as np
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd['wpe.weight'].numpy()
d=w[1:-1]-(w[:-2]+w[2:])/2
a=np.abs(d).flatten()
print("median",np.median(a),"q",np.quantile(a,[.9,.98,.99,.995,.999]))
print(np.histogram(a,bins=[0,.05,.1,.2,.3,.5,.8,1,2,4,10])[0])
# corr between adjacent rows
print("adjacent corr", np.mean([np.corrcoef(w[t,:],w[t+1,:])[0,1] for t in range(5,1000,50)]))
# column-wise mean level
print("pos 0 row rms",np.sqrt((w[0]**2).mean()))
