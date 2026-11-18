import torch, numpy as np, sys
torch.set_num_threads(1)
import torch.nn.functional as F
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
K=16
def lrmap(v,r=None):
    n,m=v.shape
    U,S,Vh=torch.linalg.svd(v,full_matrices=False)
    r=r or min(n,m)//6
    P=(U[:,:r]*S[:r])@Vh[:r]
    # map: local fraction of energy explained
    num=F.avg_pool2d((v-P)[None,None]**2,K,K)[0,0]
    den=F.avg_pool2d(v[None,None]**2,K,K)[0,0]
    return (num/den)  # high = unexplained
if __name__=="__main__":
  for k in sys.argv[1:]:
    s=lrmap(sd[k]).numpy()
    r=np.argsort(np.argsort(-s.flatten())).reshape(s.shape)/s.size
    print(k,s.shape, "median unexplained",np.median(s), "top",np.sort(s.flatten())[-5:])
    for row in r[:, :120]:
        print(''.join('#' if a<0.01 else ('+' if a<0.03 else ('.' if a<0.2 else ' ')) for a in row))
